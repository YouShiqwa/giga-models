"""Pi0.5 dual-action policy with a compact frozen-VGGT geometry stream.

This is an additive model variant.  The existing :class:`PI0Policy` remains
unchanged and keeps serving all single- and dual-action checkpoints.  This
variant adds one geometry expert beside the manipulation and mobility experts:

* frozen VGGT-Omega encodes the current left/right/wrist observations;
* a trainable convolutional reducer maps every 16x16 patch grid to 2x2;
* the resulting 12 geometry tokens join both action streams in every Pi0.5
  attention layer and may read the shared PaliGemma prefix;
* action projections, flow-matching targets, losses, and output shapes are
  inherited unchanged from the dual-action Pi0.5 policy.

The VGGT-Omega module is deliberately kept outside the registered module tree.
It is frozen, excluded from optimizers/EMA/checkpoints, and reconstructed from
the external path stored in the policy config.  This follows the deployment
pattern used by MobileManipLab's ``ThreeDQueryDMoT_ActionHeader`` stack.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import register_to_config
from torch import Tensor, nn

from .modeling_pi0 import PI0Policy, make_att_2d_masks
from .paligemma_with_expert import GemmaRMSNorm


class VGGTConvolutionalTokenReducer(nn.Module):
    """Reduce fixed per-view VGGT patch grids to compact geometry tokens.

    The implementation mirrors MobileManipLab's 3DQuery-DMoT reducer: feature
    normalization and a 1x1 projection are followed by depthwise-separable
    stride-2 convolutions.  With the default configuration, each 16x16 camera
    grid becomes a 2x2 grid, yielding four tokens per view and twelve tokens
    for the fixed three-camera RoboCasa observation.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_views: int,
        input_grid_size: int,
        output_grid_size: int,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_views <= 0:
            raise ValueError(
                'feature_dim, hidden_dim, and num_views must be positive, got '
                f'{feature_dim}, {hidden_dim}, and {num_views}.'
            )
        if output_grid_size <= 0 or input_grid_size < output_grid_size:
            raise ValueError(
                f'Expected 0 < output_grid_size <= input_grid_size, got {output_grid_size} and {input_grid_size}.'
            )
        reduction_ratio = input_grid_size // output_grid_size
        if output_grid_size * reduction_ratio != input_grid_size or reduction_ratio & (reduction_ratio - 1):
            raise ValueError(
                'VGGT convolutional reduction requires a power-of-two grid ratio, got '
                f'{input_grid_size}/{output_grid_size}.'
            )
        if norm_groups <= 0 or hidden_dim % norm_groups:
            raise ValueError(
                f'norm_groups must be a positive divisor of hidden_dim, got {norm_groups} for {hidden_dim}.'
            )

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.input_grid_size = int(input_grid_size)
        self.output_grid_size = int(output_grid_size)
        self.tokens_per_view = self.output_grid_size**2
        self.num_output_tokens = self.num_views * self.tokens_per_view

        self.input_norm = nn.LayerNorm(self.feature_dim)
        layers: list[nn.Module] = [
            nn.Conv2d(self.feature_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, self.hidden_dim),
            nn.GELU(approximate='tanh'),
        ]
        while reduction_ratio > 1:
            layers.extend(
                [
                    nn.Conv2d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=self.hidden_dim,
                        bias=False,
                    ),
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(norm_groups, self.hidden_dim),
                    nn.GELU(approximate='tanh'),
                ]
            )
            reduction_ratio //= 2
        self.convolution = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f'VGGT patch tokens must have shape [B, views*patches, feature], got {tuple(patch_tokens.shape)}.'
            )
        expected_tokens = self.num_views * self.input_grid_size**2
        expected_shape = (patch_tokens.shape[0], expected_tokens, self.feature_dim)
        if tuple(patch_tokens.shape) != expected_shape:
            raise ValueError(
                'VGGT patch tokens must contain fixed view-major square grids; '
                f'expected {expected_shape}, got {tuple(patch_tokens.shape)}.'
            )

        batch_size = patch_tokens.shape[0]
        patch_tokens = self.input_norm(patch_tokens)
        patch_grid = patch_tokens.reshape(
            batch_size,
            self.num_views,
            self.input_grid_size,
            self.input_grid_size,
            self.feature_dim,
        )
        patch_grid = patch_grid.permute(0, 1, 4, 2, 3).reshape(
            batch_size * self.num_views,
            self.feature_dim,
            self.input_grid_size,
            self.input_grid_size,
        )
        reduced_grid = self.convolution(patch_grid)
        expected_grid_shape = (
            batch_size * self.num_views,
            self.hidden_dim,
            self.output_grid_size,
            self.output_grid_size,
        )
        if tuple(reduced_grid.shape) != expected_grid_shape:
            raise RuntimeError(
                'VGGT reducer produced an unexpected grid shape: '
                f'expected {expected_grid_shape}, got {tuple(reduced_grid.shape)}.'
            )
        reduced_tokens = reduced_grid.reshape(
            batch_size,
            self.num_views,
            self.hidden_dim,
            self.tokens_per_view,
        )
        reduced_tokens = reduced_tokens.permute(0, 1, 3, 2).reshape(
            batch_size,
            self.num_output_tokens,
            self.hidden_dim,
        )
        return self.output_norm(reduced_tokens)


class FrozenVGGTOmegaPatchExtractor:
    """Lazy external VGGT-Omega patch extractor excluded from state dicts."""

    def __init__(
        self,
        *,
        repo_path: str,
        checkpoint_path: str,
        image_resolution: int,
        patch_size: int,
        feature_dim: int,
        enable_alignment: bool,
    ) -> None:
        self.repo_path = str(Path(repo_path).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.image_resolution = int(image_resolution)
        self.patch_size = int(patch_size)
        self.feature_dim = int(feature_dim)
        self.enable_alignment = bool(enable_alignment)
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        repo = Path(self.repo_path)
        checkpoint = Path(self.checkpoint_path)
        if not repo.is_dir():
            raise FileNotFoundError(f'VGGT-Omega repository not found: {repo}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'VGGT-Omega checkpoint not found: {checkpoint}')
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(enable_alignment=self.enable_alignment).eval()
        try:
            state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def to(self, device: torch.device | str) -> FrozenVGGTOmegaPatchExtractor:
        self.model.to(device=torch.device(device))
        self.model.eval()
        return self

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Extract final view-major patch states from square ``[0, 1]`` images."""
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError(f'VGGT-Omega expects images [B, 3 views, 3, H, W], got {tuple(images.shape)}.')
        device = images.device
        batch_size, num_views = images.shape[:2]
        images = images.reshape(batch_size * num_views, *images.shape[2:])
        images = F.interpolate(
            images.float(),
            size=(self.image_resolution, self.image_resolution),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        images = images.reshape(batch_size, num_views, 3, self.image_resolution, self.image_resolution)

        amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            aggregated_tokens, patch_token_start = self.model.aggregator(images)
        final_tokens = aggregated_tokens[-1]
        if final_tokens is None:
            raise RuntimeError('VGGT-Omega aggregator did not return final-layer tokens.')
        patch_tokens = final_tokens[:, :, patch_token_start:].contiguous()
        expected_patches_per_view = (self.image_resolution // self.patch_size) ** 2
        expected_shape = (batch_size, num_views, expected_patches_per_view, self.feature_dim)
        if tuple(patch_tokens.shape) != expected_shape:
            raise RuntimeError(
                'VGGT-Omega returned an unexpected patch layout: '
                f'expected {expected_shape}, got {tuple(patch_tokens.shape)}.'
            )
        return patch_tokens.reshape(batch_size, num_views * expected_patches_per_view, self.feature_dim)


def append_geometry_expert_stream(paligemma_with_expert: nn.Module) -> None:
    """Append one independent, non-time-conditioned geometry expert stream.

    Attention and FFN weights start as independent copies of the manipulation
    expert.  This gives old dual checkpoints a shape-compatible warm start,
    while the ordinary RMSNorms match the static geometry stream used by the
    ThreeDQuery-DMoT reference (only action streams receive diffusion time).
    """

    if len(paligemma_with_expert.norms) != 3:
        raise ValueError('The VGGT geometry variant requires a dual-action Pi0.5 model before adding its fourth stream.')
    for layer in paligemma_with_expert.layers:
        if len(layer.self_attn.q_proj) != 3 or len(layer.mlps) != 3:
            raise ValueError('Every Pi0.5 decoder layer must contain PaliGemma, manipulation, and mobility streams.')
        for projections in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        ):
            projections.append(copy.deepcopy(projections[1]))
        layer.mlps.append(copy.deepcopy(layer.mlps[1]))

        input_norm = layer.input_layernorms[1]
        post_attention_norm = layer.post_attention_layernorms[1]
        expert_dim = layer.mlps[1].hidden_size
        layer.input_layernorms.append(GemmaRMSNorm(expert_dim, eps=input_norm.eps, use_ada_rms_norm=False))
        layer.post_attention_layernorms.append(
            GemmaRMSNorm(expert_dim, eps=post_attention_norm.eps, use_ada_rms_norm=False)
        )

    action_norm = paligemma_with_expert.norms[1]
    expert_dim = paligemma_with_expert.layers[0].mlps[1].hidden_size
    paligemma_with_expert.norms.append(GemmaRMSNorm(expert_dim, eps=action_norm.eps, use_ada_rms_norm=False))


class PI05VGGTPolicy(PI0Policy):
    """Dual-action Pi0.5 policy with a frozen-VGGT 12-token geometry stream."""

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
        vggt_repo_path: str = '',
        vggt_checkpoint_path: str = '',
        vggt_image_resolution: int = 256,
        vggt_patch_size: int = 16,
        vggt_feature_dim: int = 2048,
        vggt_num_views: int = 3,
        vggt_output_grid_size: int = 2,
        vggt_norm_groups: int = 32,
        vggt_enable_alignment: bool = True,
        vggt_view_order: tuple[int, ...] = (0, 2, 1),
    ) -> None:
        if not pi05_enabled or not dual_action_expert:
            raise ValueError('PI05VGGTPolicy requires pi05_enabled=True and dual_action_expert=True.')
        if vggt_num_views != 3:
            raise ValueError(f'RoboCasa VGGT geometry requires exactly three views, got {vggt_num_views}.')
        if vggt_image_resolution <= 0 or vggt_patch_size <= 0 or vggt_image_resolution % vggt_patch_size:
            raise ValueError(
                'vggt_image_resolution must be a positive multiple of vggt_patch_size, got '
                f'{vggt_image_resolution} and {vggt_patch_size}.'
            )
        if tuple(sorted(vggt_view_order)) != tuple(range(vggt_num_views)):
            raise ValueError(
                f'vggt_view_order must be a permutation of [0, {vggt_num_views}), got {vggt_view_order}.'
            )
        if vggt_output_grid_size**2 != 4:
            raise ValueError(f'The geometry stream requires four tokens per view, got grid {vggt_output_grid_size}.')

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

        self.vggt_3d_enabled = True
        self.vggt_repo_path = str(vggt_repo_path)
        self.vggt_checkpoint_path = str(vggt_checkpoint_path)
        self.vggt_image_resolution = int(vggt_image_resolution)
        self.vggt_patch_size = int(vggt_patch_size)
        self.vggt_feature_dim = int(vggt_feature_dim)
        self.vggt_num_views = int(vggt_num_views)
        self.vggt_output_grid_size = int(vggt_output_grid_size)
        self.vggt_norm_groups = int(vggt_norm_groups)
        self.vggt_enable_alignment = bool(vggt_enable_alignment)
        self.vggt_view_order = tuple(vggt_view_order)
        self.vggt_input_grid_size = self.vggt_image_resolution // self.vggt_patch_size

        append_geometry_expert_stream(self.paligemma_with_expert)
        self._initialize_geometry_token_reducer()
        self.num_geometry_tokens = self.vggt_num_views * self.vggt_output_grid_size**2
        if self.num_geometry_tokens != 12:
            raise RuntimeError(f'PI05VGGTPolicy must produce 12 geometry tokens, got {self.num_geometry_tokens}.')

        # Bypass nn.Module.__setattr__: the 1B frozen backbone must not enter
        # policy state_dicts, optimizers, EMA, or FSDP parameter traversal.
        object.__setattr__(self, '_vggt_extractor', None)

    def _initialize_geometry_token_reducer(self) -> None:
        self.geometry_token_reducer = VGGTConvolutionalTokenReducer(
            feature_dim=self.vggt_feature_dim,
            hidden_dim=self.proj_width,
            num_views=self.vggt_num_views,
            input_grid_size=self.vggt_input_grid_size,
            output_grid_size=self.vggt_output_grid_size,
            norm_groups=self.vggt_norm_groups,
        )

    def _geometry_expert_parameter_names(self) -> set[str]:
        names = {
            *(f'paligemma_with_expert.norms.3.{name}' for name, _ in self.paligemma_with_expert.norms[3].named_parameters()),
        }
        for layer_index, layer in enumerate(self.paligemma_with_expert.layers):
            modules = {
                'self_attn.q_proj': layer.self_attn.q_proj[3],
                'self_attn.k_proj': layer.self_attn.k_proj[3],
                'self_attn.v_proj': layer.self_attn.v_proj[3],
                'self_attn.o_proj': layer.self_attn.o_proj[3],
                'mlps': layer.mlps[3],
                'input_layernorms': layer.input_layernorms[3],
                'post_attention_layernorms': layer.post_attention_layernorms[3],
            }
            for module_path, module in modules.items():
                names.update(
                    f'paligemma_with_expert.layers.{layer_index}.{module_path}.3.{name}'
                    for name, _ in module.named_parameters()
                )
        return names

    def _geometry_reducer_parameter_names(self) -> set[str]:
        return {f'geometry_token_reducer.{name}' for name, _ in self.geometry_token_reducer.named_parameters()}

    def initialize_geometry_expert_from_manipulation(self) -> None:
        """Recreate the static geometry expert from the loaded manipulation tower."""
        for layer in self.paligemma_with_expert.layers:
            for projections in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ):
                projections[3] = copy.deepcopy(projections[1])
            layer.mlps[3] = copy.deepcopy(layer.mlps[1])
            expert_dim = layer.mlps[1].hidden_size
            layer.input_layernorms[3] = GemmaRMSNorm(
                expert_dim,
                eps=layer.input_layernorms[1].eps,
                use_ada_rms_norm=False,
            )
            layer.post_attention_layernorms[3] = GemmaRMSNorm(
                expert_dim,
                eps=layer.post_attention_layernorms[1].eps,
                use_ada_rms_norm=False,
            )
        expert_dim = self.paligemma_with_expert.layers[0].mlps[1].hidden_size
        self.paligemma_with_expert.norms[3] = GemmaRMSNorm(
            expert_dim,
            eps=self.paligemma_with_expert.norms[1].eps,
            use_ada_rms_norm=False,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any):
        """Load native checkpoints or extend an existing Pi0.5 dual checkpoint."""
        output_loading_info = kwargs.pop('output_loading_info', False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        missing_keys = set(loading_info['missing_keys'])

        geometry_names = model._geometry_expert_parameter_names()
        missing_geometry = geometry_names & missing_keys
        if missing_geometry:
            if missing_geometry != geometry_names:
                raise RuntimeError(
                    'The checkpoint contains only part of the VGGT geometry expert; '
                    f'missing examples: {sorted(missing_geometry)[:8]}'
                )
            model.initialize_geometry_expert_from_manipulation()
            loading_info['missing_keys'] = [key for key in loading_info['missing_keys'] if key not in geometry_names]

        reducer_names = model._geometry_reducer_parameter_names()
        missing_reducer = reducer_names & missing_keys
        if missing_reducer:
            if missing_reducer != reducer_names:
                raise RuntimeError(
                    'The checkpoint contains only part of the VGGT token reducer; '
                    f'missing examples: {sorted(missing_reducer)[:8]}'
                )
            model._initialize_geometry_token_reducer()
            loading_info['missing_keys'] = [key for key in loading_info['missing_keys'] if key not in reducer_names]

        if output_loading_info:
            return model, loading_info
        return model

    def load_vggt(self, device: torch.device | str) -> FrozenVGGTOmegaPatchExtractor:
        """Materialize the external frozen VGGT-Omega model on ``device``."""
        extractor = self._vggt_extractor
        if extractor is None:
            if not self.vggt_repo_path or not self.vggt_checkpoint_path:
                raise ValueError('vggt_repo_path and vggt_checkpoint_path must be configured before loading VGGT.')
            extractor = FrozenVGGTOmegaPatchExtractor(
                repo_path=self.vggt_repo_path,
                checkpoint_path=self.vggt_checkpoint_path,
                image_resolution=self.vggt_image_resolution,
                patch_size=self.vggt_patch_size,
                feature_dim=self.vggt_feature_dim,
                enable_alignment=self.vggt_enable_alignment,
            )
            object.__setattr__(self, '_vggt_extractor', extractor)
        extractor.to(device)
        return extractor

    def train(self, mode: bool = True) -> PI05VGGTPolicy:
        super().train(mode)
        if self._vggt_extractor is not None:
            self._vggt_extractor.model.eval()
        return self

    @torch._dynamo.disable
    def _extract_vggt_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        extractor = self.load_vggt(images.device)
        return extractor(images)

    def embed_geometry(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode left/right/wrist images into 12 trainable geometry tokens."""
        if len(images) != self.vggt_num_views or len(img_masks) != self.vggt_num_views:
            raise ValueError(
                f'VGGT geometry expects {self.vggt_num_views} images and masks, got {len(images)} and {len(img_masks)}.'
            )
        ordered_images = [images[index] for index in self.vggt_view_order]
        ordered_masks = [img_masks[index] for index in self.vggt_view_order]
        batch_size = ordered_images[0].shape[0]
        if any(image.ndim != 4 or image.shape[:2] != (batch_size, 3) for image in ordered_images):
            raise ValueError('Every VGGT image must have shape [B, 3, H, W] with a shared batch size.')

        # Giga Pi0 transforms images to [-1, 1].  VGGT-Omega expects [0, 1].
        vggt_images = torch.stack(ordered_images, dim=1).float().add(1.0).mul_(0.5).clamp_(0.0, 1.0)
        patch_tokens = self._extract_vggt_patch_tokens(vggt_images).detach()
        reducer_parameter = next(self.geometry_token_reducer.parameters())
        with torch.autocast(
            device_type=patch_tokens.device.type,
            dtype=torch.bfloat16,
            enabled=patch_tokens.device.type == 'cuda',
        ):
            geometry_tokens = self.geometry_token_reducer(patch_tokens.to(dtype=reducer_parameter.dtype))
        geometry_masks = torch.stack(ordered_masks, dim=1).to(device=geometry_tokens.device, dtype=torch.bool)
        geometry_masks = geometry_masks.repeat_interleave(self.vggt_output_grid_size**2, dim=1)
        return geometry_tokens, geometry_masks

    @staticmethod
    def _append_geometry_to_suffix(
        suffix_streams: list[torch.Tensor],
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        adarms_cond: list[torch.Tensor | None],
        geometry_tokens: torch.Tensor,
        geometry_masks: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        if suffix_pad_masks.shape[0] != geometry_tokens.shape[0] or geometry_masks.shape != geometry_tokens.shape[:2]:
            raise ValueError('Geometry tokens and masks must match the action suffix batch and token dimensions.')
        suffix_streams = [*suffix_streams, geometry_tokens]
        suffix_pad_masks = torch.cat([suffix_pad_masks, geometry_masks], dim=1)
        suffix_att_masks = torch.cat([suffix_att_masks, torch.zeros_like(geometry_masks)], dim=1)
        adarms_cond = [*adarms_cond, None]
        return suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond

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
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        geometry_tokens, geometry_masks = self.embed_geometry(images, img_masks)
        suffix = self.embed_suffix(state, x_t, timestep)
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self._append_geometry_to_suffix(
            *suffix,
            geometry_tokens,
            geometry_masks,
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention_mask = make_att_2d_masks(pad_masks, att_masks)
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
        return self._project_action_outputs(outputs[1:3])

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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        geometry_tokens, geometry_masks = self.embed_geometry(images, img_masks)
        prefix_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None, None, None],
        )

        valid_action_mask = self.manipulation_action_mask + self.mobility_action_mask
        x_t = noise * valid_action_mask.to(dtype=noise.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step_with_geometry(
                state,
                prefix_pad_masks,
                past_key_values,
                geometry_tokens,
                geometry_masks,
                x_t,
                timestep.expand(batch_size),
            )
            x_t += dt * v_t
        return x_t

    def denoise_step_with_geometry(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        geometry_tokens: torch.Tensor,
        geometry_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix = self.embed_suffix(state, x_t, timestep)
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self._append_geometry_to_suffix(
            *suffix,
            geometry_tokens,
            geometry_masks,
        )

        suffix_length = suffix_pad_masks.shape[1]
        batch_size, prefix_length = prefix_pad_masks.shape
        prefix_attention_mask = prefix_pad_masks[:, None, :].expand(batch_size, suffix_length, prefix_length)
        suffix_attention_mask = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask], dim=2)
        position_ids = self._make_suffix_position_ids(prefix_pad_masks, suffix_streams)

        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, *suffix_streams],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs[1:3])


__all__ = [
    'FrozenVGGTOmegaPatchExtractor',
    'PI05VGGTPolicy',
    'VGGTConvolutionalTokenReducer',
    'append_geometry_expert_stream',
]
