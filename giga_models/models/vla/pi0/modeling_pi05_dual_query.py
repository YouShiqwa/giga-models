"""Pi0.5 dual-action policy with isolated arm/base query bridges.

This additive variant keeps the existing dual Pi0.5 action experts, action
split, flow-matching objective, and output layout unchanged.  It appends two
learned query groups to the PaliGemma prefix and changes only the structural
attention mask:

* arm/base queries read the original VLM prefix and their own bidirectional group;
* the two query groups cannot read one another;
* the arm action stream reads only arm queries from the prefix;
* the base action stream reads only base queries from the prefix;
* both action streams retain the dual policy's mutual joint attention.

Consequently, neither action expert can directly attend to image/language
tokens.  The learned queries are the only bridge from the VLM to actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.configuration_utils import register_to_config
from torch import Tensor, nn

from .modeling_pi0 import PI0Policy, make_att_2d_masks


def _validate_query_counts(arm_num_query_tokens: int, base_num_query_tokens: int) -> None:
    if arm_num_query_tokens <= 0 or base_num_query_tokens <= 0:
        raise ValueError(
            'Arm/base query counts must both be positive, got '
            f'{arm_num_query_tokens} and {base_num_query_tokens}.'
        )


def append_dual_query_pad_masks(
    raw_prefix_pad_masks: torch.Tensor,
    *,
    arm_num_query_tokens: int,
    base_num_query_tokens: int,
) -> torch.Tensor:
    """Append always-valid arm then base query positions to a prefix mask."""
    if raw_prefix_pad_masks.ndim != 2:
        raise ValueError(
            f'raw_prefix_pad_masks must have shape [B, P], got {tuple(raw_prefix_pad_masks.shape)}.'
        )
    _validate_query_counts(arm_num_query_tokens, base_num_query_tokens)
    batch_size = raw_prefix_pad_masks.shape[0]
    query_pad_masks = torch.ones(
        batch_size,
        arm_num_query_tokens + base_num_query_tokens,
        dtype=torch.bool,
        device=raw_prefix_pad_masks.device,
    )
    return torch.cat((raw_prefix_pad_masks.to(dtype=torch.bool), query_pad_masks), dim=1)


def build_dual_query_prefix_attention_mask(
    raw_prefix_pad_masks: torch.Tensor,
    raw_prefix_att_masks: torch.Tensor,
    *,
    arm_num_query_tokens: int,
    base_num_query_tokens: int,
) -> torch.Tensor:
    """Build ``[raw VLM, arm query, base query]`` prefix visibility.

    The original VLM block keeps Pi0's existing attention semantics.  Each
    query group can read every valid VLM token and every token in its own group,
    while arm/base queries are isolated in both directions.  This matches the
    original Pi0.5 prefix's bidirectional attention semantics.  ``True`` means
    that a query/key pair is visible.
    """
    if raw_prefix_pad_masks.ndim != 2 or raw_prefix_att_masks.ndim != 2:
        raise ValueError(
            'Raw prefix masks must both have shape [B, P], got '
            f'{tuple(raw_prefix_pad_masks.shape)} and {tuple(raw_prefix_att_masks.shape)}.'
        )
    if raw_prefix_pad_masks.shape != raw_prefix_att_masks.shape:
        raise ValueError(
            'Raw prefix pad/attention masks must have the same shape, got '
            f'{tuple(raw_prefix_pad_masks.shape)} and {tuple(raw_prefix_att_masks.shape)}.'
        )
    _validate_query_counts(arm_num_query_tokens, base_num_query_tokens)

    raw_prefix_pad_masks = raw_prefix_pad_masks.to(dtype=torch.bool)
    batch_size, raw_prefix_length = raw_prefix_pad_masks.shape
    total_prefix_length = raw_prefix_length + arm_num_query_tokens + base_num_query_tokens
    attention_mask = torch.zeros(
        batch_size,
        total_prefix_length,
        total_prefix_length,
        dtype=torch.bool,
        device=raw_prefix_pad_masks.device,
    )

    # Preserve the validated Pi0 image/language prefix behavior exactly.
    attention_mask[:, :raw_prefix_length, :raw_prefix_length] = make_att_2d_masks(
        raw_prefix_pad_masks,
        raw_prefix_att_masks,
    )

    arm_start = raw_prefix_length
    arm_end = arm_start + arm_num_query_tokens
    base_start = arm_end
    base_end = base_start + base_num_query_tokens

    visible_raw_keys = raw_prefix_pad_masks[:, None, :]
    attention_mask[:, arm_start:arm_end, :raw_prefix_length] = visible_raw_keys
    attention_mask[:, base_start:base_end, :raw_prefix_length] = visible_raw_keys
    attention_mask[:, arm_start:arm_end, arm_start:arm_end] = torch.ones(
        arm_num_query_tokens,
        arm_num_query_tokens,
        dtype=torch.bool,
        device=attention_mask.device,
    )
    attention_mask[:, base_start:base_end, base_start:base_end] = torch.ones(
        base_num_query_tokens,
        base_num_query_tokens,
        dtype=torch.bool,
        device=attention_mask.device,
    )
    return attention_mask


def build_dual_query_action_attention_mask(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    arm_num_query_tokens: int,
    base_num_query_tokens: int,
    arm_action_length: int,
    base_action_length: int,
) -> torch.Tensor:
    """Build cached action rows over ``[prefix, arm action, base action]``.

    Arm action rows can see only arm-query keys in the prefix, base action rows
    only base-query keys.  The suffix block itself preserves the existing Pi0.5
    dual-stream joint attention.  Raw VLM keys are deliberately all masked.
    """
    if prefix_pad_masks.ndim != 2 or suffix_pad_masks.ndim != 2 or suffix_att_masks.ndim != 2:
        raise ValueError('Prefix and suffix masks must all be rank-2 tensors.')
    if suffix_pad_masks.shape != suffix_att_masks.shape:
        raise ValueError(
            'Suffix pad/attention masks must have the same shape, got '
            f'{tuple(suffix_pad_masks.shape)} and {tuple(suffix_att_masks.shape)}.'
        )
    if prefix_pad_masks.shape[0] != suffix_pad_masks.shape[0]:
        raise ValueError('Prefix and suffix masks must use the same batch size.')
    _validate_query_counts(arm_num_query_tokens, base_num_query_tokens)
    if arm_action_length <= 0 or base_action_length <= 0:
        raise ValueError(
            f'Arm/base action lengths must both be positive, got {arm_action_length} and {base_action_length}.'
        )
    if suffix_pad_masks.shape[1] != arm_action_length + base_action_length:
        raise ValueError(
            'Suffix length must equal arm_action_length + base_action_length, got '
            f'{suffix_pad_masks.shape[1]} != {arm_action_length} + {base_action_length}.'
        )

    prefix_pad_masks = prefix_pad_masks.to(dtype=torch.bool)
    suffix_pad_masks = suffix_pad_masks.to(dtype=torch.bool)
    batch_size, prefix_length = prefix_pad_masks.shape
    raw_prefix_length = prefix_length - arm_num_query_tokens - base_num_query_tokens
    if raw_prefix_length <= 0:
        raise ValueError(
            'The prefix must contain raw VLM tokens before the two query groups, got '
            f'prefix length {prefix_length}.'
        )

    arm_query_start = raw_prefix_length
    arm_query_end = arm_query_start + arm_num_query_tokens
    base_query_start = arm_query_end
    base_query_end = base_query_start + base_num_query_tokens
    if not prefix_pad_masks[:, arm_query_start:base_query_end].all():
        raise ValueError('All appended arm/base query positions must be valid.')

    prefix_attention = torch.zeros(
        batch_size,
        suffix_pad_masks.shape[1],
        prefix_length,
        dtype=torch.bool,
        device=prefix_pad_masks.device,
    )
    prefix_attention[:, :arm_action_length, arm_query_start:arm_query_end] = (
        suffix_pad_masks[:, :arm_action_length, None]
        & prefix_pad_masks[:, None, arm_query_start:arm_query_end]
    )
    prefix_attention[:, arm_action_length:, base_query_start:base_query_end] = (
        suffix_pad_masks[:, arm_action_length:, None]
        & prefix_pad_masks[:, None, base_query_start:base_query_end]
    )

    suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    return torch.cat((prefix_attention, suffix_attention), dim=2)


def build_dual_query_full_attention_mask(
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
    """Combine prefix/query and action masks for the non-cached train pass."""
    expected_prefix_masks = append_dual_query_pad_masks(
        raw_prefix_pad_masks,
        arm_num_query_tokens=arm_num_query_tokens,
        base_num_query_tokens=base_num_query_tokens,
    )
    if not torch.equal(prefix_pad_masks.to(dtype=torch.bool), expected_prefix_masks):
        raise ValueError('prefix_pad_masks must equal raw prefix masks followed by valid arm/base queries.')

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
    batch_size, prefix_length = prefix_pad_masks.shape
    suffix_length = suffix_pad_masks.shape[1]
    full_attention = torch.zeros(
        batch_size,
        prefix_length + suffix_length,
        prefix_length + suffix_length,
        dtype=torch.bool,
        device=prefix_pad_masks.device,
    )
    full_attention[:, :prefix_length, :prefix_length] = prefix_attention
    # Prefix/query rows never read action keys; action rows are filled below.
    full_attention[:, prefix_length:, :] = action_attention
    return full_attention


class DualPI05QueryBank(nn.Module):
    """Two parameter-disjoint learned query groups in arm-then-base order."""

    def __init__(
        self,
        *,
        arm_num_query_tokens: int,
        base_num_query_tokens: int,
        hidden_dim: int,
        init_std: float,
    ) -> None:
        super().__init__()
        _validate_query_counts(arm_num_query_tokens, base_num_query_tokens)
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive, got {hidden_dim}.')
        if init_std <= 0.0:
            raise ValueError(f'init_std must be positive, got {init_std}.')

        self.arm_num_query_tokens = int(arm_num_query_tokens)
        self.base_num_query_tokens = int(base_num_query_tokens)
        self.hidden_dim = int(hidden_dim)
        self.arm_tokens = nn.Parameter(torch.empty(self.arm_num_query_tokens, self.hidden_dim))
        self.base_tokens = nn.Parameter(torch.empty(self.base_num_query_tokens, self.hidden_dim))
        nn.init.normal_(self.arm_tokens, mean=0.0, std=float(init_std))
        nn.init.normal_(self.base_tokens, mean=0.0, std=float(init_std))

    def forward(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}.')
        arm_tokens = self.arm_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        base_tokens = self.base_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return arm_tokens, base_tokens


class PI05DualQueryPolicy(PI0Policy):
    """Dual Pi0.5 whose VLM-to-action interface is a strict query bottleneck."""

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
    ) -> None:
        if not pi05_enabled or not dual_action_expert:
            raise ValueError('PI05DualQueryPolicy requires pi05_enabled=True and dual_action_expert=True.')
        _validate_query_counts(arm_num_query_tokens, base_num_query_tokens)
        if query_init_std <= 0.0:
            raise ValueError(f'query_init_std must be positive, got {query_init_std}.')

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
        )
        self.arm_num_query_tokens = int(arm_num_query_tokens)
        self.base_num_query_tokens = int(base_num_query_tokens)
        self.query_init_std = float(query_init_std)
        self.dual_query_bridge_enabled = True
        self._initialize_query_bank()

    def _initialize_query_bank(self) -> None:
        self.query_bank = DualPI05QueryBank(
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
            hidden_dim=self.paligemma_with_expert.paligemma_hidden_size,
            init_std=self.query_init_std,
        )

    def _query_parameter_names(self) -> set[str]:
        return {f'query_bank.{name}' for name, _ in self.query_bank.named_parameters()}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any):
        """Load native checkpoints or add queries to the original Pi0.5 checkpoint."""
        output_loading_info = kwargs.pop('output_loading_info', False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        query_parameter_names = model._query_parameter_names()
        missing_keys = set(loading_info['missing_keys'])
        missing_queries = query_parameter_names & missing_keys
        if missing_queries:
            if missing_queries != query_parameter_names:
                raise RuntimeError(
                    'The checkpoint contains only part of the dual query bank; '
                    f'missing keys: {sorted(missing_queries)}'
                )
            model._initialize_query_bank()
            loading_info['missing_keys'] = [
                key for key in loading_info['missing_keys'] if key not in query_parameter_names
            ]

        if output_loading_info:
            return model, loading_info
        return model

    def _append_query_embeddings(
        self,
        prefix_embs: torch.Tensor,
        raw_prefix_pad_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arm_queries, base_queries = self.query_bank(
            prefix_embs.shape[0],
            device=prefix_embs.device,
            dtype=prefix_embs.dtype,
        )
        prefix_embs = torch.cat((prefix_embs, arm_queries, base_queries), dim=1)
        prefix_pad_masks = append_dual_query_pad_masks(
            raw_prefix_pad_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
        )
        return prefix_embs, prefix_pad_masks

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
            raise RuntimeError(f'PI05DualQueryPolicy expects two action streams, got {len(suffix_streams)}.')

        attention_mask = build_dual_query_full_attention_mask(
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
        prefix_embs, prefix_pad_masks = self._append_query_embeddings(raw_prefix_embs, raw_prefix_pad_masks)
        prefix_attention_mask = build_dual_query_prefix_attention_mask(
            raw_prefix_pad_masks,
            raw_prefix_att_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=self.use_cache,
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
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(batch_size),
            )
            x_t += dt * v_t
        return x_t

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
        )
        if len(suffix_streams) != 2:
            raise RuntimeError(f'PI05DualQueryPolicy expects two action streams, got {len(suffix_streams)}.')
        attention_mask = build_dual_query_action_attention_mask(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
            arm_num_query_tokens=self.arm_num_query_tokens,
            base_num_query_tokens=self.base_num_query_tokens,
            arm_action_length=suffix_streams[0].shape[1],
            base_action_length=suffix_streams[1].shape[1],
        )
        position_ids = self._make_suffix_position_ids(prefix_pad_masks, suffix_streams)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, *suffix_streams],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs[1:])


__all__ = [
    'DualPI05QueryBank',
    'PI05DualQueryPolicy',
    'append_dual_query_pad_masks',
    'build_dual_query_action_attention_mask',
    'build_dual_query_full_attention_mask',
    'build_dual_query_prefix_attention_mask',
]
