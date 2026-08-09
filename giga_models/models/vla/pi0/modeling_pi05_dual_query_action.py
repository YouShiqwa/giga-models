"""Pi0.5 dual-query policy with branch-local query/action feedback.

This additive variant keeps the validated dual Pi0.5 action split and
flow-matching objective.  Compared with :mod:`modeling_pi05_dual_query`, each
query group can additionally read the matching noisy action stream:

* arm queries read raw VLM tokens, arm queries, and arm action tokens;
* base queries read raw VLM tokens, base queries, and base action tokens;
* arm/base query groups remain mutually invisible;
* action tokens still read only their matching queries from the prefix;
* arm/base action streams retain their existing joint attention.

Because query states now depend on the current noisy actions, only the raw VLM
prefix is static.  Inference therefore caches raw VLM K/V and recomputes the
query/action block together at every flow step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.configuration_utils import register_to_config
from torch import Tensor, nn

from .modeling_pi0 import make_att_2d_masks
from .modeling_pi05_dual_query import (
    PI05DualQueryPolicy,
    append_dual_query_pad_masks,
    build_dual_query_action_attention_mask,
    build_dual_query_prefix_attention_mask,
)
from .paligemma_with_expert import GemmaRMSNorm


_ACTION_EXPERT_INITIALIZATIONS = frozenset({'pi05', 'random'})


def build_dual_query_action_feedback_full_attention_mask(
    raw_prefix_pad_masks: torch.Tensor,
    raw_prefix_att_masks: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    arm_num_query_tokens: int,
    base_num_query_tokens: int,
    arm_action_length: int,
    base_action_length: int,
) -> torch.Tensor:
    """Build full VLM/query/action visibility with branch-local feedback.

    The flattened token order is ``[raw VLM, arm query, base query, arm
    action, base action]``.  ``True`` means that a row may read a key.
    """
    expected_prefix_masks = append_dual_query_pad_masks(
        raw_prefix_pad_masks,
        arm_num_query_tokens=arm_num_query_tokens,
        base_num_query_tokens=base_num_query_tokens,
    )
    if not torch.equal(prefix_pad_masks.to(dtype=torch.bool), expected_prefix_masks):
        raise ValueError('prefix_pad_masks must equal raw prefix masks followed by valid arm/base queries.')
    if suffix_pad_masks.ndim != 2 or suffix_att_masks.ndim != 2:
        raise ValueError('Suffix masks must both have shape [B, S].')
    if suffix_pad_masks.shape != suffix_att_masks.shape:
        raise ValueError(
            'Suffix pad/attention masks must have the same shape, got '
            f'{tuple(suffix_pad_masks.shape)} and {tuple(suffix_att_masks.shape)}.'
        )
    if suffix_pad_masks.shape[1] != arm_action_length + base_action_length:
        raise ValueError(
            'Suffix length must equal arm_action_length + base_action_length, got '
            f'{suffix_pad_masks.shape[1]} != {arm_action_length} + {base_action_length}.'
        )

    prefix_attention = build_dual_query_prefix_attention_mask(
        raw_prefix_pad_masks,
        raw_prefix_att_masks,
        arm_num_query_tokens=arm_num_query_tokens,
        base_num_query_tokens=base_num_query_tokens,
    )
    action_attention = build_dual_query_action_attention_mask(
        prefix_pad_masks,
        suffix_pad_masks,
        suffix_att_masks,
        arm_num_query_tokens=arm_num_query_tokens,
        base_num_query_tokens=base_num_query_tokens,
        arm_action_length=arm_action_length,
        base_action_length=base_action_length,
    )

    batch_size, raw_prefix_length = raw_prefix_pad_masks.shape
    prefix_length = prefix_pad_masks.shape[1]
    suffix_length = suffix_pad_masks.shape[1]
    full_attention = torch.zeros(
        batch_size,
        prefix_length + suffix_length,
        prefix_length + suffix_length,
        dtype=torch.bool,
        device=raw_prefix_pad_masks.device,
    )
    full_attention[:, :prefix_length, :prefix_length] = prefix_attention
    full_attention[:, prefix_length:, :] = action_attention

    arm_query_start = raw_prefix_length
    arm_query_end = arm_query_start + arm_num_query_tokens
    base_query_start = arm_query_end
    base_query_end = base_query_start + base_num_query_tokens
    arm_action_start = prefix_length
    arm_action_end = arm_action_start + arm_action_length
    base_action_start = arm_action_end
    base_action_end = base_action_start + base_action_length

    # Query/action feedback stays branch-local.  Raw VLM rows remain independent
    # of all dynamic tokens, so their K/V can be cached exactly once.
    full_attention[:, arm_query_start:arm_query_end, arm_action_start:arm_action_end] = (
        suffix_pad_masks[:, None, :arm_action_length]
    )
    full_attention[:, base_query_start:base_query_end, base_action_start:base_action_end] = (
        suffix_pad_masks[:, None, arm_action_length:]
    )
    return full_attention


def build_dual_query_action_feedback_cached_attention_mask(
    raw_prefix_pad_masks: torch.Tensor,
    raw_prefix_att_masks: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    arm_num_query_tokens: int,
    base_num_query_tokens: int,
    arm_action_length: int,
    base_action_length: int,
) -> torch.Tensor:
    """Return dynamic rows over ``[cached raw VLM, queries, actions]``."""
    full_attention = build_dual_query_action_feedback_full_attention_mask(
        raw_prefix_pad_masks,
        raw_prefix_att_masks,
        prefix_pad_masks,
        suffix_pad_masks,
        suffix_att_masks,
        arm_num_query_tokens=arm_num_query_tokens,
        base_num_query_tokens=base_num_query_tokens,
        arm_action_length=arm_action_length,
        base_action_length=base_action_length,
    )
    return full_attention[:, raw_prefix_pad_masks.shape[1] :, :]


def _reset_linear_tree(module: nn.Module) -> None:
    """Restore PyTorch initialization while preserving AdaRMS zero gates."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            child.reset_parameters()
    for child in module.modules():
        if not isinstance(child, GemmaRMSNorm):
            continue
        if child.use_ada_rms_norm:
            nn.init.zeros_(child.dense.weight)
        else:
            nn.init.zeros_(child.weight)


class PI05DualQueryActionPolicy(PI05DualQueryPolicy):
    """Dual 16-query Pi0.5 with matching query/action bidirectional attention."""

    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        n_action_steps: int = 50,
        num_steps: int = 10,
        use_cache: bool = True,
        pi05_enabled: bool = True,
        dual_action_expert: bool = True,
        manipulation_action_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        mobility_action_indices: tuple[int, ...] = (7, 8, 9, 10, 11),
        manipulation_loss_weight: float = 0.5,
        mobility_loss_weight: float = 0.5,
        arm_num_query_tokens: int = 16,
        base_num_query_tokens: int = 16,
        query_init_std: float = 0.02,
        action_expert_initialization: str = 'pi05',
        action_expert_random_seed: int = 6666,
    ) -> None:
        if not use_cache:
            raise ValueError('PI05DualQueryActionPolicy requires use_cache=True for raw-VLM-only caching.')
        if action_expert_initialization not in _ACTION_EXPERT_INITIALIZATIONS:
            raise ValueError(
                'action_expert_initialization must be one of '
                f'{sorted(_ACTION_EXPERT_INITIALIZATIONS)}, got {action_expert_initialization!r}.'
            )
        super().__init__(
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            proj_width=proj_width,
            n_action_steps=n_action_steps,
            num_steps=num_steps,
            use_cache=use_cache,
            pi05_enabled=pi05_enabled,
            dual_action_expert=dual_action_expert,
            manipulation_action_indices=manipulation_action_indices,
            mobility_action_indices=mobility_action_indices,
            manipulation_loss_weight=manipulation_loss_weight,
            mobility_loss_weight=mobility_loss_weight,
            arm_num_query_tokens=arm_num_query_tokens,
            base_num_query_tokens=base_num_query_tokens,
            query_init_std=query_init_std,
        )
        self.action_expert_initialization = str(action_expert_initialization)
        self.action_expert_random_seed = int(action_expert_random_seed)
        self.dual_query_action_attention_enabled = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any):
        """Load a checkpoint and optionally reset only the two action heads.

        ``reset_action_experts_after_load`` is deliberately a load-time flag,
        not a saved architecture option.  Native trained checkpoints therefore
        never erase their learned action weights when loaded for evaluation.
        """
        reset_action_experts = bool(kwargs.pop('reset_action_experts_after_load', False))
        output_loading_info = kwargs.pop('output_loading_info', False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        if reset_action_experts:
            if model.action_expert_initialization != 'random':
                raise ValueError(
                    'reset_action_experts_after_load=True requires '
                    "action_expert_initialization='random'."
                )
            model.reset_action_experts(model.action_expert_random_seed)

        if output_loading_info:
            return model, loading_info
        return model

    def reset_action_experts(self, seed: int) -> None:
        """Randomize both complete action paths while leaving VLM/query intact."""
        cuda_devices = sorted(
            {
                parameter.device.index
                for parameter in self.parameters()
                if parameter.is_cuda and parameter.device.index is not None
            }
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))

            for module in (
                self.action_in_proj,
                self.action_out_proj,
                self.mobility_action_in_proj,
                self.mobility_action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ):
                _reset_linear_tree(module)

            for layer in self.paligemma_with_expert.layers:
                for stream_index in (1, 2):
                    for projections in (
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                        layer.self_attn.o_proj,
                    ):
                        _reset_linear_tree(projections[stream_index])
                    _reset_linear_tree(layer.mlps[stream_index])
                    _reset_linear_tree(layer.input_layernorms[stream_index])
                    _reset_linear_tree(layer.post_attention_layernorms[stream_index])

            for stream_index in (1, 2):
                _reset_linear_tree(self.paligemma_with_expert.norms[stream_index])

    def _make_dynamic_position_ids(
        self,
        raw_prefix_pad_masks: torch.Tensor,
        suffix_streams: list[torch.Tensor],
    ) -> torch.Tensor:
        raw_offsets = torch.sum(raw_prefix_pad_masks, dim=-1)[:, None]
        query_count = self.arm_num_query_tokens + self.base_num_query_tokens
        query_positions = torch.arange(
            query_count,
            device=raw_prefix_pad_masks.device,
            dtype=raw_offsets.dtype,
        )[None, :]
        query_positions = raw_offsets + query_positions

        action_offsets = raw_offsets + query_count
        action_positions = []
        for stream in suffix_streams:
            stream_positions = torch.arange(
                stream.shape[1],
                device=stream.device,
                dtype=action_offsets.dtype,
            )[None, :]
            action_positions.append(action_offsets + stream_positions)
        return torch.cat((query_positions, *action_positions), dim=1)

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tensor:
        raw_prefix_embs, raw_prefix_pad_masks, raw_prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_embs, prefix_pad_masks = self._append_query_embeddings(raw_prefix_embs, raw_prefix_pad_masks)
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
        )
        if len(suffix_streams) != 2:
            raise RuntimeError(f'PI05DualQueryActionPolicy expects two action streams, got {len(suffix_streams)}.')

        attention_mask = build_dual_query_action_feedback_full_attention_mask(
            raw_prefix_pad_masks,
            raw_prefix_att_masks,
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
            arm_action_length=suffix_streams[0].shape[1],
            base_action_length=suffix_streams[1].shape[1],
        )
        position_ids = self._make_full_position_ids(prefix_pad_masks, suffix_streams)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, *suffix_streams],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs[1:])

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        if noise is None:
            noise = self.sample_noise((batch_size, self.n_action_steps, self.max_action_dim), device)

        raw_prefix_embs, raw_prefix_pad_masks, raw_prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        raw_attention_mask = make_att_2d_masks(raw_prefix_pad_masks, raw_prefix_att_masks)
        raw_position_ids = torch.cumsum(raw_prefix_pad_masks, dim=1) - 1
        _, raw_past_key_values = self.paligemma_with_expert.forward(
            attention_mask=raw_attention_mask,
            position_ids=raw_position_ids,
            past_key_values=None,
            inputs_embeds=[raw_prefix_embs, None, None],
            use_cache=True,
            fill_kv_cache=True,
            adarms_cond=[None, None, None],
        )

        valid_action_mask = self.manipulation_action_mask + self.mobility_action_mask
        x_t = noise * valid_action_mask.to(dtype=noise.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step(
                state,
                raw_prefix_pad_masks,
                raw_prefix_att_masks,
                raw_past_key_values,
                x_t,
                timestep.expand(batch_size),
            )
            x_t += dt * v_t
        return x_t

    def denoise_step(
        self,
        state: torch.Tensor,
        raw_prefix_pad_masks: torch.Tensor,
        raw_prefix_att_masks: torch.Tensor,
        raw_past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        arm_queries, base_queries = self.query_bank(
            x_t.shape[0],
            device=x_t.device,
            dtype=self.paligemma_with_expert.embed_tokens.weight.dtype,
        )
        query_embs = torch.cat((arm_queries, base_queries), dim=1)
        prefix_pad_masks = append_dual_query_pad_masks(
            raw_prefix_pad_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
        )
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
        )
        if len(suffix_streams) != 2:
            raise RuntimeError(f'PI05DualQueryActionPolicy expects two action streams, got {len(suffix_streams)}.')

        attention_mask = build_dual_query_action_feedback_cached_attention_mask(
            raw_prefix_pad_masks,
            raw_prefix_att_masks,
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
            arm_action_length=suffix_streams[0].shape[1],
            base_action_length=suffix_streams[1].shape[1],
        )
        position_ids = self._make_dynamic_position_ids(raw_prefix_pad_masks, suffix_streams)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=raw_past_key_values,
            inputs_embeds=[query_embs, *suffix_streams],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs[1:])


__all__ = [
    'PI05DualQueryActionPolicy',
    'build_dual_query_action_feedback_cached_attention_mask',
    'build_dual_query_action_feedback_full_attention_mask',
]
