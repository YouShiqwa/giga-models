import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipMultiheadAttentionPoolingHead, SiglipVisionEmbeddings
from transformers.utils import auto_docstring, can_return_tuple


def get_transformers_siglip_vision_config() -> SiglipVisionConfig:
    return CONFIG_MAPPING['siglip_vision_model'](
        hidden_size=1152,
        intermediate_size=4304,
        num_channels=3,
        num_attention_heads=16,
        num_hidden_layers=27,
        num_image_tokens=256,
        patch_size=14,
        projection_dim=2048,
        projector_hidden_act='gelu_fast',
        torch_dtype='float32',
        vision_use_head=False,
    )


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_ada_rms_norm: bool = False):
        super().__init__()
        self.eps = eps
        self.use_ada_rms_norm = use_ada_rms_norm
        if use_ada_rms_norm:
            self.dense = nn.Linear(dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, cond: torch.Tensor | None = None):
        normed_inputs = self._norm(x.float())

        if self.use_ada_rms_norm:
            modulation = self.dense(cond)
            scale, shift, gate = torch.chunk(modulation.unsqueeze(1), 3, dim=-1)
            normed_inputs = normed_inputs.float() * (1.0 + scale.float()) + shift.float()
            return normed_inputs.type_as(x), gate.type_as(x)

        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = normed_inputs * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        if self.use_ada_rms_norm:
            return f'{tuple(self.dense.weight.shape)}, eps={self.eps}, use_ada_rms_norm=True'
        else:
            return f'{tuple(self.weight.shape)}, eps={self.eps}'


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.config._attn_implementation = 'sdpa'
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, 'vision_use_head') else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        """Forward pass of the SigLIP vision encoder.

        Args:
            pixel_values: Image tensor expected by SigLIP (B, C, H, W).
            output_attentions: Whether to return attention maps.
            output_hidden_states: Whether to return hidden states.
            interpolate_pos_encoding: Enable pos-encoding interpolation for different sizes.

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and optionally pooled output.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = hidden_states.to(dtype=torch.bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoder_outputs: BaseModelOutput = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )


# Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaMultiModalProjector
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int = 1152, projection_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to the transformer hidden size."""
        hidden_states = self.linear(image_features)
        return hidden_states


class RoPEEmbedding(nn.Module):
    """Precomputed RoPE embeddings for improved performance.

    This implementation precomputes sin/cos values for a maximum sequence length, avoiding redundant trigonometric calculations during forward passes.
    """

    def __init__(self, dim: int, max_wavelength: int = 10_000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_wavelength = max_wavelength
        self.max_seq_len = max_seq_len

        # Precompute frequency exponents and inverse frequencies
        d_half = dim // 2
        freq_exponents = (2.0 / dim) * torch.arange(d_half, dtype=torch.float32)
        inv_freq = 1.0 / (max_wavelength**freq_exponents)

        # Precompute sin and cos for all positions up to max_seq_len
        # Shape: [max_seq_len, d_half]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, d_half]

        # Precompute sin and cos values
        # We expand to [max_seq_len, 1, d_half] for broadcasting in forward
        cos_cached = torch.cos(freqs).unsqueeze(1)  # [max_seq_len, 1, d_half]
        sin_cached = torch.sin(freqs).unsqueeze(1)  # [max_seq_len, 1, d_half]

        # Register as buffers so they automatically move to the correct device with the model
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
        """Applies RoPE positions [B, L] to x [B, L, H, D].

        Args:
            x: Input tensor of shape [B, L, H, D]
            positions: Position indices of shape [B, L]

        Returns:
            Rotated tensor of shape [B, L, H, D]
        """
        dtype = x.dtype
        x = x.to(torch.float32)

        # Index precomputed sin/cos values using positions
        # positions: [B, L] -> cos/sin: [B, L, 1, d_half]
        cos = self.cos_cached[positions]  # [B, L, 1, d_half]
        sin = self.sin_cached[positions]  # [B, L, 1, d_half]

        # Apply rotary embeddings
        d_half = self.dim // 2
        x1, x2 = x.split(d_half, dim=-1)  # Each: [B, L, H, d_half]

        # Rotate: out1 = x1 * cos - x2 * sin, out2 = x2 * cos + x1 * sin
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin

        return res.to(dtype)


class GemmaAttentionWithExpert(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        # PaliGemma params
        paligemma_hidden_size: int = 2048,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 1,
        paligemma_head_dim: int = 256,
        paligemma_attention_bias: bool = False,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 1,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        # RoPE params
        rope_max_wavelength: int = 10_000,
        rope_max_seq_len: int = 8192,
        dual_action_expert: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_action_experts = 2 if dual_action_expert else 1

        action_q_proj = nn.Linear(
            expert_hidden_size,
            expert_num_attention_heads * expert_head_dim,
            bias=expert_attention_bias,
        )
        action_k_proj = nn.Linear(
            expert_hidden_size,
            expert_num_key_value_heads * expert_head_dim,
            bias=expert_attention_bias,
        )
        action_v_proj = nn.Linear(
            expert_hidden_size,
            expert_num_key_value_heads * expert_head_dim,
            bias=expert_attention_bias,
        )
        action_o_proj = nn.Linear(
            expert_num_attention_heads * expert_head_dim,
            expert_hidden_size,
            bias=expert_attention_bias,
        )
        self.q_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_attention_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                action_q_proj,
                *([copy.deepcopy(action_q_proj)] if dual_action_expert else []),
            ]
        )
        self.k_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_key_value_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                action_k_proj,
                *([copy.deepcopy(action_k_proj)] if dual_action_expert else []),
            ]
        )
        self.v_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_key_value_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                action_v_proj,
                *([copy.deepcopy(action_v_proj)] if dual_action_expert else []),
            ]
        )
        self.o_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_num_attention_heads * paligemma_head_dim, paligemma_hidden_size, bias=paligemma_attention_bias),
                action_o_proj,
                *([copy.deepcopy(action_o_proj)] if dual_action_expert else []),
            ]
        )

        self.paligemma_num_attention_heads = paligemma_num_attention_heads
        self.paligemma_num_key_value_heads = paligemma_num_key_value_heads
        self.paligemma_head_dim = paligemma_head_dim
        self.expert_num_attention_heads = expert_num_attention_heads
        self.expert_num_key_value_heads = expert_num_key_value_heads
        self.expert_head_dim = expert_head_dim

        assert paligemma_head_dim == expert_head_dim
        assert paligemma_num_attention_heads == expert_num_attention_heads
        assert paligemma_num_key_value_heads == expert_num_key_value_heads
        self.rope_embedding = RoPEEmbedding(dim=paligemma_head_dim, max_wavelength=rope_max_wavelength, max_seq_len=rope_max_seq_len)

    def initialize_mobility_expert_from_manipulation(self) -> None:
        """Copy the original action attention weights into the mobility branch."""
        if self.num_action_experts != 2:
            raise RuntimeError('Mobility initialization requires dual_action_expert=True.')
        for projections in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            projections[2] = copy.deepcopy(projections[1])

    def forward(
        self,
        inputs_embeds: List[Optional[torch.Tensor]],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Optional[dict] = None,
        fill_kv_cache: bool = False,
    ) -> List[Optional[torch.Tensor]]:
        """Joint attention over PaliGemma and one or two action streams.

        Args:
            inputs_embeds: ``[paligemma, manipulation, mobility?]`` streams.
            position_ids: (B, L) rotary positions.
            attention_mask: (B, L, L) attention mask.
            use_cache: Whether to use KV cache.
            past_key_values: Optional cache dict per layer.
            fill_kv_cache: If True, fill cache; otherwise, append to it.

        Returns:
            List[Optional[Tensor]]: outputs per stream aligned to inputs order.
        """
        query_states = []
        key_states = []
        value_states = []

        if len(inputs_embeds) != len(self.q_proj):
            raise ValueError(f'Expected {len(self.q_proj)} input streams, but got {len(inputs_embeds)}.')

        for stream_idx, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            input_shape = hidden_states.shape[:-1]
            head_dim = self.paligemma_head_dim if stream_idx == 0 else self.expert_head_dim
            hidden_shape = (*input_shape, -1, head_dim)
            query_states.append(self.q_proj[stream_idx](hidden_states).view(hidden_shape))
            key_states.append(self.k_proj[stream_idx](hidden_states).view(hidden_shape))
            value_states.append(self.v_proj[stream_idx](hidden_states).view(hidden_shape))

        if not query_states:
            raise ValueError('At least one attention input stream must be present.')

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        query_states = self.rope_embedding(query_states, position_ids)
        key_states = self.rope_embedding(key_states, position_ids)

        if use_cache:
            if fill_kv_cache:
                past_key_values[self.layer_idx] = {
                    'key_states': key_states,
                    'value_states': value_states,
                }
            else:
                key_states = torch.cat([past_key_values[self.layer_idx]['key_states'], key_states], dim=1)
                value_states = torch.cat([past_key_values[self.layer_idx]['value_states'], value_states], dim=1)

        num_att_heads = self.paligemma_num_attention_heads  # Assume same for both
        num_key_value_heads = self.paligemma_num_key_value_heads
        head_dim = self.paligemma_head_dim
        batch_size = query_states.shape[0]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if num_key_value_heads != num_att_heads:
            # key_states: (B, num_kv_heads, L, D) -> (B, num_att_heads, L, D)
            key_states = torch.repeat_interleave(key_states, num_att_heads // num_key_value_heads, dim=1)
            value_states = torch.repeat_interleave(value_states, num_att_heads // num_key_value_heads, dim=1)

        att_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask[:, None, :, :],
            is_causal=False,
        )
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_att_heads * head_dim)

        outputs_embeds: list[Optional[torch.Tensor]] = []
        start = 0
        for stream_idx, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            if att_output.dtype != self.o_proj[stream_idx].weight.dtype:
                att_output_i = att_output[:, start:end].to(self.o_proj[stream_idx].weight.dtype)
            else:
                att_output_i = att_output[:, start:end]
            outputs_embeds.append(self.o_proj[stream_idx](att_output_i))
            start = end

        return outputs_embeds


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, hidden_act: str = 'gelu_pytorch_tanh'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated MLP block used in both streams."""
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class GemmaDecoderLayerWithExpert(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        pi05_enabled: bool,
        # PaliGemma params
        paligemma_hidden_size: int = 2048,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 1,
        paligemma_head_dim: int = 256,
        paligemma_attention_bias: bool = False,
        paligemma_intermediate_size: int = 16384,
        paligemma_hidden_act: str = 'gelu_pytorch_tanh',
        paligemma_rms_norm_eps: float = 1e-6,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 1,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        expert_intermediate_size: int = 4096,
        expert_hidden_act: str = 'gelu_pytorch_tanh',
        expert_rms_norm_eps: float = 1e-6,
        # RoPE params
        rope_max_wavelength: int = 10_000,
        rope_max_seq_len: int = 8192,
        dual_action_expert: bool = False,
    ):
        super().__init__()
        self.num_action_experts = 2 if dual_action_expert else 1
        self.self_attn = GemmaAttentionWithExpert(
            layer_idx,
            paligemma_hidden_size,
            paligemma_num_attention_heads,
            paligemma_num_key_value_heads,
            paligemma_head_dim,
            paligemma_attention_bias,
            expert_hidden_size,
            expert_num_attention_heads,
            expert_num_key_value_heads,
            expert_head_dim,
            expert_attention_bias,
            rope_max_wavelength,
            rope_max_seq_len,
            dual_action_expert,
        )

        action_mlp = GemmaMLP(expert_hidden_size, expert_intermediate_size, expert_hidden_act)
        self.mlps = nn.ModuleList(
            [
                GemmaMLP(paligemma_hidden_size, paligemma_intermediate_size, paligemma_hidden_act),
                action_mlp,
                *([copy.deepcopy(action_mlp)] if dual_action_expert else []),
            ]
        )

        action_input_layernorm = GemmaRMSNorm(expert_hidden_size, eps=expert_rms_norm_eps, use_ada_rms_norm=pi05_enabled)
        self.input_layernorms = nn.ModuleList(
            [
                GemmaRMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                action_input_layernorm,
                *([copy.deepcopy(action_input_layernorm)] if dual_action_expert else []),
            ]
        )
        action_post_attention_layernorm = GemmaRMSNorm(
            expert_hidden_size,
            eps=expert_rms_norm_eps,
            use_ada_rms_norm=pi05_enabled,
        )
        self.post_attention_layernorms = nn.ModuleList(
            [
                GemmaRMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                action_post_attention_layernorm,
                *([copy.deepcopy(action_post_attention_layernorm)] if dual_action_expert else []),
            ]
        )

        self.pi05_enabled = pi05_enabled

    def initialize_mobility_expert_from_manipulation(self) -> None:
        """Copy all per-layer manipulation parameters into the mobility expert."""
        if self.num_action_experts != 2:
            raise RuntimeError('Mobility initialization requires dual_action_expert=True.')
        self.self_attn.initialize_mobility_expert_from_manipulation()
        self.mlps[2] = copy.deepcopy(self.mlps[1])
        self.input_layernorms[2] = copy.deepcopy(self.input_layernorms[1])
        self.post_attention_layernorms[2] = copy.deepcopy(self.post_attention_layernorms[1])

    def gated_residual(self, x, y, gate):
        if x is None or y is None:
            return None
        if gate is None:
            return x + y
        return x + y * gate

    def forward(
        self,
        inputs_embeds: List[Optional[torch.Tensor]],
        adarms_cond: List[Optional[torch.Tensor]],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Optional[dict] = None,
        fill_kv_cache: bool = False,
    ) -> List[Optional[torch.Tensor]]:
        """Decoder layer with joint multi-stream attention and optional AdaRMS.

        Args:
            inputs_embeds: ``[paligemma, manipulation, mobility?]`` embeds.
            adarms_cond: Optional conditioning vectors for AdaRMS.
            position_ids: (B, L) positions for RoPE.
            attention_mask: (B, L, L) attention mask.
            use_cache: Whether to use KV cache.
            past_key_values: Optional cache dict.
            fill_kv_cache: Whether to fill or reuse KV cache.

        Returns:
            List[Optional[Tensor]]: Updated hidden states per stream.
        """
        residuals = list(inputs_embeds)
        normed_embeds = []
        attn_gates = []

        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.pi05_enabled and adarms_cond[i] is not None:
                    normed_h, attn_gate = self.input_layernorms[i](hidden_states, adarms_cond[i])
                    normed_embeds.append(normed_h)
                    attn_gates.append(attn_gate)
                else:
                    normed_embeds.append(self.input_layernorms[i](hidden_states))
                    attn_gates.append(None)
            else:
                normed_embeds.append(None)
                attn_gates.append(None)

        attn_outputs = self.self_attn(normed_embeds, position_ids, attention_mask, use_cache, past_key_values, fill_kv_cache)

        after_attn_embeds = []
        for i, (residual, attn_output, attn_gate) in enumerate(zip(residuals, attn_outputs, attn_gates)):
            if residual is not None:
                after_attn_embeds.append(self.gated_residual(residual, attn_output, attn_gate))
            else:
                after_attn_embeds.append(None)

        outputs = []
        for i, hidden_states in enumerate(after_attn_embeds):
            if hidden_states is not None:
                residual = hidden_states
                if self.pi05_enabled and adarms_cond[i] is not None:
                    normed_h, mlp_gate = self.post_attention_layernorms[i](hidden_states, adarms_cond[i])
                else:
                    normed_h = self.post_attention_layernorms[i](hidden_states)
                    mlp_gate = None

                mlp_out = self.mlps[i](normed_h)
                outputs.append(self.gated_residual(residual, mlp_out, mlp_gate))
            else:
                outputs.append(None)

        return outputs


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        pi05_enabled: bool = False,
        dual_action_expert: bool = False,
        # Paligemma params
        paligemma_vocab_size: int = 257152,
        paligemma_pad_token_id: int = 0,
        paligemma_num_hidden_layers: int = 18,
        paligemma_hidden_size: int = 2048,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 1,
        paligemma_attention_bias: bool = False,
        paligemma_intermediate_size: int = 16384,
        paligemma_hidden_act: str = 'gelu_pytorch_tanh',
        paligemma_rms_norm_eps: float = 1e-6,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 1,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        expert_intermediate_size: int = 4096,
        expert_hidden_act: str = 'gelu_pytorch_tanh',
        expert_rms_norm_eps: float = 1e-6,
        # RoPE params
        rope_max_wavelength: int = 10_000,
        rope_max_seq_len: int = 8192,
    ):
        super().__init__()
        self.pi05_enabled = pi05_enabled
        self.dual_action_expert = dual_action_expert
        self.num_action_experts = 2 if dual_action_expert else 1

        siglip_vision_config = get_transformers_siglip_vision_config()

        # Vision and projection
        self.vision_tower = SiglipVisionTransformer(siglip_vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=siglip_vision_config.hidden_size, projection_dim=siglip_vision_config.projection_dim
        )
        self.paligemma_hidden_size = paligemma_hidden_size

        # Language embed
        self.embed_tokens = nn.Embedding(paligemma_vocab_size, paligemma_hidden_size, paligemma_pad_token_id)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayerWithExpert(
                    layer_idx=i,
                    pi05_enabled=pi05_enabled,
                    paligemma_hidden_size=paligemma_hidden_size,
                    paligemma_num_attention_heads=paligemma_num_attention_heads,
                    paligemma_num_key_value_heads=paligemma_num_key_value_heads,
                    paligemma_head_dim=paligemma_hidden_size // paligemma_num_attention_heads,
                    paligemma_attention_bias=paligemma_attention_bias,  # gemma default
                    paligemma_intermediate_size=paligemma_intermediate_size,
                    paligemma_hidden_act=paligemma_hidden_act,
                    paligemma_rms_norm_eps=paligemma_rms_norm_eps,  # gemma default
                    expert_hidden_size=expert_hidden_size,
                    expert_num_attention_heads=expert_num_attention_heads,
                    expert_num_key_value_heads=expert_num_key_value_heads,
                    expert_head_dim=expert_head_dim,
                    expert_attention_bias=expert_attention_bias,
                    expert_intermediate_size=expert_intermediate_size,
                    expert_hidden_act=expert_hidden_act,
                    expert_rms_norm_eps=expert_rms_norm_eps,
                    rope_max_wavelength=rope_max_wavelength,
                    rope_max_seq_len=rope_max_seq_len,
                    dual_action_expert=dual_action_expert,
                )
                for i in range(paligemma_num_hidden_layers)
            ]
        )

        # Final norms
        action_norm = GemmaRMSNorm(expert_hidden_size, eps=expert_rms_norm_eps, use_ada_rms_norm=pi05_enabled)
        self.norms = nn.ModuleList(
            [
                GemmaRMSNorm(paligemma_hidden_size, eps=1e-6),
                action_norm,
                *([copy.deepcopy(action_norm)] if dual_action_expert else []),
            ]
        )

    def initialize_mobility_expert_from_manipulation(self) -> None:
        """Copy the complete pretrained manipulation tower into mobility."""
        if not self.dual_action_expert:
            raise RuntimeError('Mobility initialization requires dual_action_expert=True.')
        for layer in self.layers:
            layer.initialize_mobility_expert_from_manipulation()
        self.norms[2] = copy.deepcopy(self.norms[1])

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP and project to hidden size."""
        image_outputs = self.vision_tower(image)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token ids into continuous vectors."""
        return self.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        adarms_cond: List[torch.FloatTensor] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], dict]:
        """Run the stacked joint-stream decoder with optional caching and AdaRMS.

        Args:
            attention_mask: (B, L, L) attention mask for both streams.
            position_ids: (B, L) RoPE positions.
            past_key_values: Optional KV cache dict to reuse.
            inputs_embeds: ``[paligemma, manipulation, mobility?]``.
            use_cache: Whether to use KV cache.
            fill_kv_cache: If True, populate cache from inputs.
            adarms_cond: Optional per-stream modulation vectors for AdaRMS.

        Returns:
            (outputs_embeds, past_key_values): outputs per stream and the KV cache.
        """
        inputs_embeds = [input_embed.to(dtype=torch.bfloat16) if input_embed is not None else None for input_embed in inputs_embeds]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if use_cache and past_key_values is None:
                past_key_values = {}

            hidden_states_list = inputs_embeds
            for layer in self.layers:
                hidden_states_list = layer(
                    hidden_states_list,
                    adarms_cond=adarms_cond,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    fill_kv_cache=fill_kv_cache,
                )

            outputs_embeds = []
            for i, hidden_states in enumerate(hidden_states_list):
                if hidden_states is not None:
                    if self.pi05_enabled and adarms_cond[i] is not None:
                        out_emb, _ = self.norms[i](hidden_states, adarms_cond[i])
                    else:
                        out_emb = self.norms[i](hidden_states)
                    outputs_embeds.append(out_emb)
                else:
                    outputs_embeds.append(None)

            return outputs_embeds, past_key_values
