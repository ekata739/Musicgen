
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from xformers import ops

from .rope import RotaryEmbedding
from .streaming import StreamingModule

_efficient_attention_backend: str = 'torch'

def set_efficient_attention_backend(backend: str = 'torch'):
    global _efficient_attention_backend
    assert backend in ['xformers', 'torch']
    _efficient_attention_backend = backend

def _get_attention_time_dimension(memory_efficient: bool) -> int:
    if _efficient_attention_backend == 'torch' and memory_efficient:
        return 2
    else:
        return 1 #Returns the time axis index depending on the backend layout.

def _is_profiled() -> bool:
    try:
        from xformers.profiler import profiler
    except ImportError: #returns whether xformers profiler is actively running
        return False
    return profiler._Profiler._CURRENT_PROFILER is not None

def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    if norm_type == 'layer_norm':#Returns the specified normalization module
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert dim % 2 == 0 #Creates sinusoidal positional embeddings (like those used in the original Transformer). Each position is mapped to a unique sine/cosine value.
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

def expand_repeated_kv(x: torch.Tensor, n_rep: int, memory_efficient: bool) -> torch.Tensor:
    if n_rep == 1: #Handles key-value duplication for multi-query attention where kv_repeat > 1.
        return x
    if _efficient_attention_backend == 'torch' and memory_efficient:
        bs, n_kv_heads, slen, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
        )
    else:
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool = True,
                 device=None, dtype=None):
        super().__init__() #Applies a learned scaling factor to the residual connection. Helps stabilize training in deep networks.
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full((channels,), init, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__() #Instead of fixed sin/cos positions, this learns an embedding for relative positions between tokens. Helpful in music or time-series tasks where relative position matters more than absolute.
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, embed_dim) #t maps relative distances between tokens to embedding vectors.
        coords = torch.arange(max_len)
        rel_coords = coords[:, None] - coords[None, :] #This creates a matrix of relative distances between every pair of positions.
        rel_coords += max_len - 1 #Since relative distances can be negative, we shift them to become valid positive indices into
        self.register_buffer('rel_coords', rel_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape #Applies the relative position bias to every token in the sequence.
        rel_pos = self.rel_pos_emb(self.rel_coords[:seq_len, :seq_len])
        x = x + rel_pos.unsqueeze(0) #adds batch dimension for broadcasting.
        return x

class MultiScaleHeterogeneousAdapter(nn.Module):
    """Multi-Scale Temporal Adapter with Single Scale, no tasks."""
    def __init__(self, embed_dim: int, num_heads: int, scale: int,
                 device=None, dtype=None):
        super().__init__()
        self.scale = scale #How long the learned "prompt" should be—controls temporal range.
        self.head_dim = embed_dim // num_heads #Dimension per attention head.

        # Initialize prompt with sinusoidal pattern for classical melodies
        self.prompt = nn.Parameter(
            torch.sin(torch.linspace(0, 2 * torch.pi, scale)).unsqueeze(-1).repeat(1, self.head_dim) * 0.1
        ) #Creates a [scale, head_dim] matrix of sinusoidal signals (like musical waveforms). Sinusoidal patterns can capture natural periodic structure


        # These act as learned memory templates, initialized from a sinusoidal base.
        self.gate = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * 0.5,
            requires_grad=True
        ) #A scalar gate (sigmoid(self.gate)) controls how much of the adapter's output influences the final result.

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, seq_len, num_heads, head_dim = q.shape #Input: Q/K/V from main attention head.
        adapter_output = torch.zeros_like(q) #Output: same shape as q.


        scale_prompt = self.prompt #Match the length of the prompt to current sequence length.
        if scale_prompt.size(0) > seq_len:
            scale_prompt = scale_prompt[:seq_len]
        elif scale_prompt.size(0) < seq_len:
            scale_prompt = F.pad(scale_prompt, (0, 0, 0, seq_len - scale_prompt.size(0)))
        scale_prompt_expanded = scale_prompt.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_heads, -1) #Reshape prompt to shape: (batch, seq_len, num_heads, head_dim) so it can be attended to.
        attn = F.scaled_dot_product_attention(
            q, scale_prompt_expanded, scale_prompt_expanded, dropout_p=0.0
        ) #Perform attention over the prompt, not the original key/value inputs.
#This returns context from the fixed sinusoidal memory.
        gate = torch.sigmoid(self.gate)
        adapter_output += gate * attn

        return adapter_output #Modulate output via gate, so the model can learn how much to use this adapter.

class StreamingMultiheadAttention(StreamingModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None, custom: bool = False,
                 memory_efficient: bool = False, attention_as_float32: bool = False,
                 rope: tp.Optional[RotaryEmbedding] = None, cross_attention: bool = False,
                 safe_streaming: bool = True, qk_layer_norm: bool = False, kv_repeat: int = 1,
                 use_adapter: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if past_context is not None:
            assert causal

        self.embed_dim = embed_dim
        self.causal = causal
        self.past_context = past_context
        self.memory_efficient = memory_efficient
        self.attention_as_float32 = attention_as_float32
        self.rope = rope
        self.cross_attention = cross_attention
        self.safe_streaming = safe_streaming
        self.num_heads = num_heads
        self.dropout = dropout
        self.kv_repeat = kv_repeat
        if cross_attention:
            assert not causal
            assert rope is None

        if memory_efficient:
            _verify_xformers_memory_efficient_compat()

        self.custom = _is_custom(custom, memory_efficient)
        if self.custom:
            out_dim = embed_dim
            assert num_heads % kv_repeat == 0
            assert not cross_attention or kv_repeat == 1
            num_kv = num_heads // kv_repeat
            kv_dim = (embed_dim // num_heads) * num_kv
            out_dim += 2 * kv_dim
            in_proj = nn.Linear(embed_dim, out_dim, bias=bias, **factory_kwargs)
            self.in_proj_weight = in_proj.weight
            self.in_proj_bias = in_proj.bias
            if bias:
                self.in_proj_bias.data.zero_()
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            if bias:
                self.out_proj.bias.data.zero_()
        else:
            assert not qk_layer_norm
            assert kv_repeat == 1
            self.mha = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True,
                **factory_kwargs)
        self.qk_layer_norm = qk_layer_norm
        if qk_layer_norm:
            assert self.custom
            assert kv_repeat == 1
            ln_dim = embed_dim
            self.q_layer_norm = nn.LayerNorm(ln_dim)
            self.k_layer_norm = nn.LayerNorm(ln_dim)

        self.use_adapter = use_adapter
        self.adapters = nn.ModuleList()
        if self.use_adapter:
            # Adapter 1: Micro-patterns (short scale)
            self.adapters.append(MultiScaleHeterogeneousAdapter(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=150,
                device=device,
                dtype=dtype
            ))
            # Adapter 2: Melody (short-medium scale)
            self.adapters.append(MultiScaleHeterogeneousAdapter(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=400,
                device=device,
                dtype=dtype
            ))
            # Adapter 3: Harmony (medium scale)
            self.adapters.append(MultiScaleHeterogeneousAdapter(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=600,
                device=device,
                dtype=dtype
            ))
            # Adapter 4: Sectional structure (long scale)
            self.adapters.append(MultiScaleHeterogeneousAdapter(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=1000,
                device=device,
                dtype=dtype
            ))
            # Adapter 5: Full structure (longest scale)
            self.adapters.append(MultiScaleHeterogeneousAdapter(
                embed_dim=embed_dim,
                num_heads=num_heads,
                scale=1500,
                device=device,
                dtype=dtype
            ))
            # Weights for combining adapter outputs, prioritizing melody
            self.adapter_weights = nn.Parameter(
                torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20], device=device, dtype=dtype),
                requires_grad=True
            )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not self.custom:
            keys = [n for n, _ in self.mha.named_parameters()]
            for key in keys:
                if prefix + key in state_dict:
                    state_dict[prefix + "mha." + key] = state_dict.pop(prefix + key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_mask(self, current_steps: int, device: torch.device, dtype: torch.dtype):
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        if self.memory_efficient:
            from xformers.ops import LowerTriangularMask
            if current_steps == 1:
                return None
            elif 'past_keys' in self._streaming_state:
                raise RuntimeError("Not supported at the moment")
            else:
                return LowerTriangularMask()
        if self._streaming_state:
            past_steps = self._streaming_state['past_keys'].shape[time_dim]
        else:
            past_steps = 0

        queries_pos = torch.arange(past_steps, current_steps + past_steps, device=device).view(-1, 1)
        keys_pos = torch.arange(past_steps + current_steps, device=device).view(1, -1)
        delta = queries_pos - keys_pos
        valid = delta >= 0
        if self.past_context is not None:
            valid &= (delta <= self.past_context)
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def _complete_kv(self, k, v):
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        if self.cross_attention:
            return k, v
        if self._streaming_state:
            pk = self._streaming_state['past_keys']
            nk = torch.cat([pk, k], dim=time_dim)
            if v is k:
                nv = nk
            else:
                pv = self._streaming_state['past_values']
                nv = torch.cat([pv, v], dim=time_dim)
        else:
            nk = k
            nv = v

        assert nk.shape[time_dim] == nv.shape[time_dim]
        offset = 0
        if self.past_context is not None:
            offset = max(0, nk.shape[time_dim] - self.past_context)
        if self._is_streaming:
            self._streaming_state['past_keys'] = nk[:, offset:]
            if v is not k:
                self._streaming_state['past_values'] = nv[:, offset:]
            if 'offset' in self._streaming_state:
                self._streaming_state['offset'] += offset
            else:
                self._streaming_state['offset'] = torch.tensor(0)
        return nk, nv

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor):
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        assert self.rope is not None
        if 'past_keys' in self._streaming_state:
            past_keys_offset = self._streaming_state['past_keys'].shape[1]
        else:
            past_keys_offset = 0
        if 'offset' in self._streaming_state:
            past_context_offset = int(self._streaming_state['offset'].item())
        else:
            past_context_offset = 0
        streaming_offset = past_context_offset + past_keys_offset
        return self.rope.rotate_qk(query, key, start=streaming_offset, time_dim=time_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False, prompt_desc: tp.Optional[str] = None):
        assert not is_causal
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        layout = "b h t d" if time_dim == 2 else "b t h d"
        dtype = query.dtype
        if self._is_streaming:
            assert self.causal or self.cross_attention

        custom_attn_mask = attn_mask is not None

        if self.causal:
            assert attn_mask is None
            assert query.shape[1] == key.shape[1]
            assert value.shape[1] == key.shape[1]
            attn_mask = self._get_mask(query.shape[1], query.device, query.dtype)

        if self.custom:
            assert need_weights is False
            assert key_padding_mask is None
            if self.cross_attention:
                dim = self.in_proj_weight.shape[0] // 3
                bias_q = self.in_proj_bias[:dim] if self.in_proj_bias is not None else None
                bias_k = self.in_proj_bias[dim: 2 * dim] if self.in_proj_bias is not None else None
                bias_v = self.in_proj_bias[2 * dim:] if self.in_proj_bias is not None else None
                q = nn.functional.linear(query, self.in_proj_weight[:dim], bias_q)
                k = nn.functional.linear(key, self.in_proj_weight[dim: 2 * dim], bias_k)
                v = nn.functional.linear(value, self.in_proj_weight[2 * dim:], bias_v)
                if self.qk_layer_norm:
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                q, k, v = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k, v]]
            else:
                assert query is key
                assert value is key
                projected = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
                if self.kv_repeat == 1:
                    bound_layout = "b h p t d" if time_dim == 2 else "b t p h d"
                    packed = rearrange(projected, f"b t (p h d) -> {bound_layout}", p=3, h=self.num_heads)
                    q, k, v = ops.unbind(packed, dim=2)
                else:
                    embed_dim = self.embed_dim
                    per_head_dim = (embed_dim // self.num_heads)
                    kv_heads = self.num_heads // kv_repeat
                    q = projected[:, :, :embed_dim]
                    start = embed_dim
                    end = start + per_head_dim * kv_heads
                    k = projected[:, :, start:end]
                    v = projected[:, :, end:]
                    q = rearrange(q, f"b t (h d) -> {layout}", h=self.num_heads)
                    k = rearrange(k, f"b t (h d) -> {layout}", h=kv_heads)
                    v = rearrange(v, f"b t (h d) -> {layout}", h=kv_heads)

                if self.qk_layer_norm:
                    assert self.kv_repeat == 1
                    q, k = [rearrange(x, f"{layout} -> b t (h d)") for x in [q, k]]
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                    q, k = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k]]
                if self.rope:
                    q, k = self._apply_rope(q, k)
                k, v = self._complete_kv(k, v)
                if self.kv_repeat > 1:
                    k = expand_repeated_kv(k, self.kv_repeat, self.memory_efficient)
                    v = expand_repeated_kv(v, self.kv_repeat, self.memory_efficient)

            if self.attention_as_float32:
                q, k, v = [x.float() for x in [q, k, v]]
            if self.memory_efficient:
                if custom_attn_mask:
                    seq_len = query.shape[1]
                    attn_mask = attn_mask.to(q.dtype)
                    attn_mask = attn_mask.repeat((q.shape[0], 1, 1, 1))
                    attn_mask = attn_mask[..., :seq_len, :seq_len]

                p = self.dropout if self.training else 0
                if _efficient_attention_backend == 'torch':
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=attn_mask is not None, dropout_p=p)
                else:
                    x = ops.memory_efficient_attention(q, k, v, attn_mask, p=p)
            else:
                q = q / q.shape[-1] ** 0.5
                key_layout = layout.replace('t', 'k')
                query_layout = layout
                if self._is_streaming and self.safe_streaming and q.device.type == 'cuda':
                    with torch.autocast(device_type=q.device.type, dtype=torch.float32):
                        pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                else:
                    pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                if attn_mask is not None:
                    pre_w = pre_w + attn_mask
                w = torch.softmax(pre_w, dim=-1)
                w = F.dropout(w, self.dropout, training=self.training).to(v)
                x = torch.einsum(f"b h t k, {key_layout} -> {layout}", w, v)

            if self.use_adapter and self.adapters:
                adapter_outputs = [adapter(q, k, v) for adapter in self.adapters]
                weights = F.softmax(self.adapter_weights, dim=0)
                adapter_output = sum(w * out for w, out in zip(weights, adapter_outputs))
                x = x + adapter_output

            x = x.to(dtype)
            x = rearrange(x, f"{layout} -> b t (h d)", h=self.num_heads)
            x = self.out_proj(x)
        else:
            key, value = self._complete_kv(key, value)
            if self.attention_as_float32:
                query, key, value = [x.float() for x in [query, key, value]]
            x, _ = self.mha(
                query, key, value, key_padding_mask,
                need_weights, attn_mask, average_attn_weights)
            x = x.to(dtype)

        return x, None

class StreamingTransformerLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 bias_ff: bool = True, bias_attn: bool = True, causal: bool = False,
                 past_context: tp.Optional[int] = None, custom: bool = False,
                 memory_efficient: bool = False, attention_as_float32: bool = False,
                 qk_layer_norm: bool = False, qk_layer_norm_cross: bool = False,
                 cross_attention: bool = False, layer_scale: tp.Optional[float] = None,
                 rope: tp.Optional[RotaryEmbedding] = None, attention_dropout: tp.Optional[float] = None,
                 kv_repeat: int = 1, norm: str = 'layer_norm', use_adapter: bool = True,
                 device=None, dtype=None, **kwargs):
        super().__init__(d_model, num_heads, dim_feedforward, dropout,
                         device=device, dtype=dtype, batch_first=True, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        attn_kwargs = {
            'embed_dim': d_model,
            'num_heads': num_heads,
            'dropout': dropout if attention_dropout is None else attention_dropout,
            'bias': bias_attn,
            'custom': custom,
            'memory_efficient': memory_efficient,
            'attention_as_float32': attention_as_float32,
            'use_adapter': use_adapter,
        }
        self.self_attn = StreamingMultiheadAttention(
            causal=causal, past_context=past_context, rope=rope, qk_layer_norm=qk_layer_norm,
            kv_repeat=kv_repeat, **attn_kwargs, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias_ff, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias_ff, **factory_kwargs)

        self.layer_scale_1 = nn.Identity() if layer_scale is None else LayerScale(d_model, layer_scale, **factory_kwargs)
        self.layer_scale_2 = nn.Identity() if layer_scale is None else LayerScale(d_model, layer_scale, **factory_kwargs)

        self.cross_attention = None
        if cross_attention:
            self.cross_attention = StreamingMultiheadAttention(
                cross_attention=True, qk_layer_norm=qk_layer_norm_cross,
                **attn_kwargs, **factory_kwargs)
            self.dropout_cross = nn.Dropout(dropout)
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
            self.layer_scale_cross = nn.Identity() if layer_scale is None else LayerScale(d_model, layer_scale, **factory_kwargs)
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)

    def _sa_block(self, x: torch.Tensor, attn_mask: tp.Optional[torch.Tensor],
                  key_padding_mask: tp.Optional[torch.Tensor], prompt_desc: tp.Optional[str] = None) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           prompt_desc=prompt_desc)[0]
        return self.dropout1(x)

    def _cross_attention_block(self, src: torch.Tensor, cross_attention_src: torch.Tensor) -> torch.Tensor:
        assert self.cross_attention is not None
        x = self.cross_attention(src, cross_attention_src, cross_attention_src, need_weights=False)[0]
        return self.dropout_cross(x)

    def forward(self, src: torch.Tensor, src_mask: tp.Optional[torch.Tensor] = None,
                src_key_padding_mask: tp.Optional[torch.Tensor] = None,
                cross_attention_src: tp.Optional[torch.Tensor] = None,
                prompt_desc: tp.Optional[str] = None):
        if self.cross_attention is None:
            assert cross_attention_src is None
        else:
            assert cross_attention_src is not None
        x = src
        if self.norm_first:
            x = x + self.layer_scale_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, prompt_desc=prompt_desc))
            if cross_attention_src is not None:
                x = x + self.layer_scale_cross(
                    self._cross_attention_block(self.norm_cross(x), cross_attention_src))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(x + self.layer_scale_1(
                self._sa_block(x, src_mask, src_key_padding_mask, prompt_desc=prompt_desc)))
            if cross_attention_src is not None:
                x = self.norm_cross(
                    x + self.layer_scale_cross(self._cross_attention_block(src, cross_attention_src)))
            x = self.norm2(x + self.layer_scale_2(self._ff_block(x)))
        return x

class StreamingTransformer(StreamingModule):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, bias_ff: bool = True, bias_attn: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None,
                 custom: bool = False, memory_efficient: bool = False, attention_as_float32: bool = False,
                 cross_attention: bool = False, layer_scale: tp.Optional[float] = None,
                 positional_embedding: str = 'sin', max_period: float = 10_000, positional_scale: float = 1.,
                 xpos: bool = False, lr: tp.Optional[float] = None, weight_decay: tp.Optional[float] = None,
                 layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
                 checkpointing: str = 'none', use_adapter: bool = True, device=None, dtype=None, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.weight_decay = weight_decay
        self.lr = lr

        assert positional_embedding in ['sin', 'rope', 'sin_rope', 'relative']
        self.rope = None
        self.rel_pos_enc = None
        if self.positional_embedding in ['rope', 'sin_rope']:
            assert _is_custom(custom, memory_efficient)
            self.rope = RotaryEmbedding(d_model // num_heads, max_period=max_period,
                                        xpos=xpos, scale=positional_scale, device=device)
        elif self.positional_embedding == 'relative':
            self.rel_pos_enc = RelativePositionalEncoding(d_model, max_len=max_period)

        self.checkpointing = checkpointing
        assert checkpointing in ['none', 'torch', 'xformers_default', 'xformers_mm']
        if self.checkpointing.startswith('xformers'):
            _verify_xformers_internal_compat()

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                    dropout=dropout, bias_ff=bias_ff, bias_attn=bias_attn,
                    causal=causal, past_context=past_context, custom=custom,
                    memory_efficient=memory_efficient, attention_as_float32=attention_as_float32,
                    cross_attention=cross_attention, layer_scale=layer_scale, rope=self.rope,
                    use_adapter=use_adapter, device=device, dtype=dtype, **kwargs))

        if self.checkpointing != 'none':
            for layer in self.layers:
                layer._magma_checkpointed = True

    def _apply_layer(self, layer, *args, **kwargs):
        method = self.checkpointing
        if method == 'none':
            return layer(*args, **kwargs)
        elif method == 'torch':
            return torch_checkpoint(layer, *args, use_reentrant=False, **kwargs)
        elif method.startswith('xformers'):
            from xformers.checkpoint_fairinternal import checkpoint, _get_default_policy
            if method == 'xformers_default':
                allow_list = [
                    "xformers.efficient_attention_forward_cutlass.default",
                    "xformers_flash.flash_fwd.default",
                    "aten.addmm.default",
                    "aten.mm.default",
                ]
            elif method == 'xformers_mm':
                allow_list = ["aten.addmm.default", "aten.mm.default"]
            else:
                raise ValueError(f"xformers checkpointing xformers policy {method} is not known.")
            policy_fn = _get_default_policy(allow_list)
            return checkpoint(layer, *args, policy_fn=policy_fn, **kwargs)
        else:
            raise ValueError(f"Checkpointing method {method} is unknown.")

    def forward(self, x: torch.Tensor, prompt_desc: tp.Optional[str] = None, *args, **kwargs):
        B, T, C = x.shape

        if 'offsets' in self._streaming_state:
            offsets = self._streaming_state['offsets']
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)

        if self.positional_embedding == 'relative':
            x = self.rel_pos_enc(x)
        elif self.positional_embedding in ['sin', 'sin_rope']:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = self._apply_layer(layer, x, prompt_desc=prompt_desc, *args, **kwargs)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return x

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        if self.weight_decay is not None:
            group["weight_decay"] = self.weight_decay
        return group

def _verify_xformers_memory_efficient_compat():
    try:
        from xformers.ops import memory_efficient_attention, LowerTriangularMask
    except ImportError:
        raise ImportError("xformers not installed.")

def _verify_xformers_internal_compat():
    try:
        from xformers.checkpoint_fairinternal import checkpoint, _get_default_policy
    except ImportError:
        raise ImportError("Francisco's fairinternal xformers not installed.")

def _is_custom(custom: bool, memory_efficient: bool):
    return custom or memory_efficient
