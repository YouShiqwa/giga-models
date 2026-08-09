import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn

from .paligemma_with_expert import PaliGemmaWithExpertModel


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Mps is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == 'mps' and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device: str | torch.device = 'cpu'
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar
    positions."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      pad_masks: bool[B, N] indicating valid (true) vs. padding (false) tokens.
      att_masks: int[B, N] defining attention type. A `1` at a position
                 indicates the start of a new causal block.

    Returns:
        A 2D boolean attention mask of shape (B, N, N).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class PI0Policy(ModelMixin, ConfigMixin):
    """pi0: A Vision-Language-Action Flow Model for General Robot Control.

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        n_action_steps: int = 50,
        num_steps: int = 10,
        use_cache: bool = True,
        pi05_enabled: bool = False,
        dual_action_expert: bool = False,
        manipulation_action_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        mobility_action_indices: tuple[int, ...] = (7, 8, 9, 10, 11),
        manipulation_loss_weight: float = 0.5,
        mobility_loss_weight: float = 0.5,
    ):
        super().__init__()

        manipulation_action_indices = tuple(manipulation_action_indices)
        mobility_action_indices = tuple(mobility_action_indices)
        if dual_action_expert and not pi05_enabled:
            raise ValueError('dual_action_expert is only supported for Pi0.5.')
        if dual_action_expert:
            self._validate_action_branches(
                max_action_dim,
                manipulation_action_indices,
                mobility_action_indices,
                manipulation_loss_weight,
                mobility_loss_weight,
            )

        # Store the parameters
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.proj_width = proj_width
        self.n_action_steps = n_action_steps
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.pi05_enabled = pi05_enabled
        self.dual_action_expert = dual_action_expert
        self.manipulation_action_indices = manipulation_action_indices
        self.mobility_action_indices = mobility_action_indices
        self.manipulation_loss_weight = manipulation_loss_weight
        self.mobility_loss_weight = mobility_loss_weight

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            pi05_enabled=pi05_enabled,
            dual_action_expert=dual_action_expert,
        )

        # Projections are float32
        if self.pi05_enabled:
            self.time_mlp_in = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
            self.time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
        else:
            self.state_proj = nn.Linear(self.max_state_dim, self.proj_width, dtype=torch.float32)
            self.action_time_mlp_in = nn.Linear(self.proj_width * 2, self.proj_width, dtype=torch.float32)
            self.action_time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)

        self.action_in_proj = nn.Linear(self.max_action_dim, self.proj_width, dtype=torch.float32)
        self.action_out_proj = nn.Linear(self.proj_width, self.max_action_dim, dtype=torch.float32)
        if self.dual_action_expert:
            self.mobility_action_in_proj = copy.deepcopy(self.action_in_proj)
            self.mobility_action_out_proj = copy.deepcopy(self.action_out_proj)

        manipulation_mask = torch.zeros(self.max_action_dim, dtype=torch.float32)
        mobility_mask = torch.zeros(self.max_action_dim, dtype=torch.float32)
        if self.dual_action_expert:
            manipulation_mask[list(self.manipulation_action_indices)] = 1.0
            mobility_mask[list(self.mobility_action_indices)] = 1.0
        else:
            manipulation_mask.fill_(1.0)
        self.register_buffer('manipulation_action_mask', manipulation_mask, persistent=False)
        self.register_buffer('mobility_action_mask', mobility_mask, persistent=False)

    @staticmethod
    def _validate_action_branches(
        max_action_dim: int,
        manipulation_action_indices: tuple[int, ...],
        mobility_action_indices: tuple[int, ...],
        manipulation_loss_weight: float,
        mobility_loss_weight: float,
    ) -> None:
        if not manipulation_action_indices or not mobility_action_indices:
            raise ValueError('Both action branches must contain at least one channel.')
        manipulation_set = set(manipulation_action_indices)
        mobility_set = set(mobility_action_indices)
        if len(manipulation_set) != len(manipulation_action_indices) or len(mobility_set) != len(mobility_action_indices):
            raise ValueError('Action branch indices must not contain duplicates.')
        overlap = manipulation_set & mobility_set
        if overlap:
            raise ValueError(f'Action branch indices must be disjoint, but overlap at {sorted(overlap)}.')
        invalid = sorted(index for index in manipulation_set | mobility_set if index < 0 or index >= max_action_dim)
        if invalid:
            raise ValueError(f'Action branch indices must be in [0, {max_action_dim}), but got {invalid}.')
        if manipulation_loss_weight < 0 or mobility_loss_weight < 0:
            raise ValueError('Action branch loss weights must be non-negative.')
        if manipulation_loss_weight + mobility_loss_weight <= 0:
            raise ValueError('At least one action branch loss weight must be positive.')

    def _mobility_parameter_names(self) -> set[str]:
        """Return parameters introduced exclusively by the mobility tower."""
        if not self.dual_action_expert:
            return set()

        names = {
            *(f'mobility_action_in_proj.{name}' for name, _ in self.mobility_action_in_proj.named_parameters()),
            *(f'mobility_action_out_proj.{name}' for name, _ in self.mobility_action_out_proj.named_parameters()),
            *(f'paligemma_with_expert.norms.2.{name}' for name, _ in self.paligemma_with_expert.norms[2].named_parameters()),
        }
        for layer_idx, layer in enumerate(self.paligemma_with_expert.layers):
            modules = {
                'self_attn.q_proj': layer.self_attn.q_proj[2],
                'self_attn.k_proj': layer.self_attn.k_proj[2],
                'self_attn.v_proj': layer.self_attn.v_proj[2],
                'self_attn.o_proj': layer.self_attn.o_proj[2],
                'mlps': layer.mlps[2],
                'input_layernorms': layer.input_layernorms[2],
                'post_attention_layernorms': layer.post_attention_layernorms[2],
            }
            for module_path, module in modules.items():
                names.update(
                    f'paligemma_with_expert.layers.{layer_idx}.{module_path}.2.{name}' for name, _ in module.named_parameters()
                )
        return names

    def initialize_mobility_expert_from_manipulation(self) -> None:
        """Clone the complete pretrained action expert into the mobility tower."""
        if not self.dual_action_expert:
            raise RuntimeError('Mobility initialization requires dual_action_expert=True.')
        self.paligemma_with_expert.initialize_mobility_expert_from_manipulation()
        self.mobility_action_in_proj = copy.deepcopy(self.action_in_proj)
        self.mobility_action_out_proj = copy.deepcopy(self.action_out_proj)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load old and dual-tower checkpoints with automatic expert cloning.

        When an old single-action-expert checkpoint is loaded with
        ``dual_action_expert=True``, all missing mobility parameters are copied
        from the loaded action expert. Native dual-tower checkpoints load both
        branches without reinitialization.
        """
        output_loading_info = kwargs.pop('output_loading_info', False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        if model.dual_action_expert:
            mobility_parameter_names = model._mobility_parameter_names()
            missing_keys = set(loading_info['missing_keys'])
            missing_mobility = mobility_parameter_names & missing_keys
            if missing_mobility:
                if missing_mobility != mobility_parameter_names:
                    missing_preview = sorted(missing_mobility)[:8]
                    raise RuntimeError(
                        'The checkpoint contains only part of the mobility expert. '
                        f'Refusing an ambiguous initialization; missing examples: {missing_preview}'
                    )
                model.initialize_mobility_expert_from_manipulation()
                loading_info['missing_keys'] = [key for key in loading_info['missing_keys'] if key not in mobility_parameter_names]

        if output_loading_info:
            return model, loading_info
        return model

    def _make_suffix_position_ids(
        self,
        prefix_pad_masks: torch.Tensor,
        suffix_streams: list[torch.Tensor],
    ) -> torch.Tensor:
        """Create action positions, aligning both experts at each horizon step."""
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = []
        for stream in suffix_streams:
            stream_positions = torch.arange(stream.shape[1], device=stream.device, dtype=prefix_offsets.dtype)[None, :]
            position_ids.append(prefix_offsets + stream_positions)
        return torch.cat(position_ids, dim=1)

    def _make_full_position_ids(
        self,
        prefix_pad_masks: torch.Tensor,
        suffix_streams: list[torch.Tensor],
    ) -> torch.Tensor:
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        suffix_position_ids = self._make_suffix_position_ids(prefix_pad_masks, suffix_streams)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

    def _project_action_outputs(self, action_outputs: list[torch.Tensor]) -> torch.Tensor:
        manipulation_output = action_outputs[0][:, -self.n_action_steps :]
        manipulation_output = manipulation_output.to(dtype=self.action_out_proj.weight.dtype)
        manipulation_prediction = self.action_out_proj(manipulation_output)
        if not self.dual_action_expert:
            return manipulation_prediction

        mobility_output = action_outputs[1][:, -self.n_action_steps :]
        mobility_output = mobility_output.to(dtype=self.mobility_action_out_proj.weight.dtype)
        mobility_prediction = self.mobility_action_out_proj(mobility_output)
        manipulation_mask = self.manipulation_action_mask.to(dtype=manipulation_prediction.dtype)
        mobility_mask = self.mobility_action_mask.to(dtype=mobility_prediction.dtype)
        return manipulation_prediction * manipulation_mask + mobility_prediction * mobility_mask

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
        """Full forward pass for one diffusion denoising step.

        Args:
            images: List of image tensors, each shaped (B, C, H, W) after batching.
            img_masks: List of boolean masks corresponding to images, each (B,).
            lang_tokens: Language token ids (B, L).
            lang_masks: Language attention mask (B, L) with True for valid tokens.
            state: State tensor (B, state_dim) if pi05 is disabled else ignored.
            x_t: Noisy action tokens (B, n_action_steps, action_dim).
            timestep: Diffusion timestep as float tensor (B,).

        Returns:
            Predicted v_t with shape (B, n_action_steps, action_dim).
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = self._make_full_position_ids(prefix_pad_masks, suffix_streams)

        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, *suffix_streams],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs[1:])

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """Generate Gaussian noise for the action trajectory.

        Args:
            shape: Desired output shape, typically (B, n_action_steps, action_dim).
            device: Target device string or torch.device.

        Returns:
            A float32 tensor of standard normal samples with the given shape.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def embed_prefix(
        self, images: list[torch.Tensor], img_masks: list[torch.Tensor], lang_tokens: torch.Tensor, lang_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed visual and language inputs as the transformer prefix.

        Args:
            images: List of (B, C, H, W) tensors.
            img_masks: List of (B,) boolean masks for image presence.
            lang_tokens: (B, L) token ids.
            lang_masks: (B, L) boolean mask; True indicates valid tokens.

        Returns:
            A tuple of (embs, pad_masks, att_masks):
              - embs: (B, Np, D) concatenated image and language embeddings
              - pad_masks: (B, Np) valid token mask
              - att_masks: (B, Np) attention mask scheme selector
        """
        # Optimize: batch process images and pre-allocate tensors
        num_images = len(images)

        # Stack images and masks for batch processing
        images_stacked = torch.stack(images, dim=0)  # (num_images, bsize, ...)
        img_masks_stacked = torch.stack(img_masks, dim=0)  # (num_images, bsize)

        # Batch embed all images at once
        # Reshape to (num_images * bsize, ...)
        orig_shape = images_stacked.shape
        images_flat = images_stacked.reshape(-1, *orig_shape[2:])
        img_embs_flat = self.paligemma_with_expert.embed_image(images_flat)

        # Reshape back to (num_images, bsize, num_img_embs, emb_dim)
        bsize = orig_shape[1]
        img_embs = img_embs_flat.reshape(num_images, bsize, *img_embs_flat.shape[1:])

        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        num_img_embs = img_embs.shape[2]

        # Expand masks: (num_images, bsize) -> (num_images, bsize, num_img_embs)
        img_masks_expanded = img_masks_stacked[:, :, None].expand(num_images, bsize, num_img_embs)

        # Reshape to (bsize, num_images * num_img_embs, emb_dim)
        img_embs_concat = img_embs.transpose(0, 1).reshape(bsize, num_images * num_img_embs, img_emb_dim)
        img_masks_concat = img_masks_expanded.transpose(0, 1).reshape(bsize, num_images * num_img_embs)

        # Process language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_emb = lang_emb.to(dtype=img_embs_concat.dtype)

        num_lang_embs = lang_emb.shape[1]
        total_seq_len = num_images * num_img_embs + num_lang_embs

        # Pre-allocate final tensors
        embs = torch.empty(bsize, total_seq_len, img_emb_dim, dtype=img_embs_concat.dtype, device=img_embs_concat.device)
        pad_masks = torch.empty(bsize, total_seq_len, dtype=torch.bool, device=img_embs_concat.device)

        # Fill pre-allocated tensors
        embs[:, : num_images * num_img_embs] = img_embs_concat
        embs[:, num_images * num_img_embs :] = lang_emb
        pad_masks[:, : num_images * num_img_embs] = img_masks_concat
        pad_masks[:, num_images * num_img_embs :] = lang_masks

        # Create attention masks (all zeros for full attention between image and language)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, total_seq_len)

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        """Embed state, action and time tokens into one or two expert streams.

        Args:
            state: (B, state_dim) robot state; ignored when pi05 is enabled.
            noisy_actions: (B, n_action_steps, action_dim) current x_t.
            timestep: (B,) diffusion time in [0, 1].

        Returns:
            ``(streams, pad_masks, att_masks, conditions)``. In dual mode the
            streams are manipulation and mobility, each with ``T`` tokens; all
            ``2T`` action tokens form one mutually visible attention block.
        """
        if noisy_actions.shape[-1] != self.max_action_dim:
            raise ValueError(
                f'Expected noisy actions with {self.max_action_dim} channels, but got shape {tuple(noisy_actions.shape)}.'
            )

        manipulation_input = noisy_actions
        if self.dual_action_expert:
            manipulation_input = manipulation_input * self.manipulation_action_mask.to(dtype=noisy_actions.dtype)
        manipulation_emb = self.action_in_proj(manipulation_input)
        bsize = manipulation_emb.shape[0]
        dtype = manipulation_emb.dtype
        device = manipulation_emb.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device)
        time_emb = time_emb.type(dtype=dtype)

        if self.pi05_enabled:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            adarms_cond = time_emb
        else:
            # Fuse timestep + action information using an MLP
            time_emb = time_emb[:, None, :].expand_as(manipulation_emb)
            action_time_emb = torch.cat([manipulation_emb, time_emb], dim=2)

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            manipulation_emb = action_time_emb
            adarms_cond = None

        if not self.pi05_enabled:
            state_emb = self.state_proj(state)
            manipulation_emb = torch.cat([state_emb[:, None, :], manipulation_emb], dim=1)

        streams = [manipulation_emb]
        conditions: list[torch.Tensor | None] = [adarms_cond]
        if self.dual_action_expert:
            mobility_input = noisy_actions * self.mobility_action_mask.to(dtype=noisy_actions.dtype)
            streams.append(self.mobility_action_in_proj(mobility_input))
            conditions.append(adarms_cond)

        total_suffix_len = sum(stream.shape[1] for stream in streams)
        pad_masks = torch.ones(bsize, total_suffix_len, dtype=torch.bool, device=device)
        # One causal-block marker makes both noisy action branches mutually visible.
        att_masks = torch.zeros(bsize, total_suffix_len, dtype=torch.bool, device=device)
        att_masks[:, 0] = True
        if not self.pi05_enabled:
            # Preserve Pi0's separate state block followed by the action block.
            att_masks[:, 1] = True
        return streams, pad_masks, att_masks, conditions

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
        """Run the full inference loop to predict an action trajectory.

        Args:
            images: List of (B, C, H, W) image tensors.
            img_masks: List of (B,) boolean masks.
            lang_tokens: (B, L) token ids.
            lang_masks: (B, L) boolean mask for tokens.
            state: (B, state_dim) robot state.
            noise: Optional initial noise; if None, generated internally.

        Returns:
            Predicted actions with shape (B, n_action_steps, action_dim).
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.n_action_steps, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs] + [None] * self.paligemma_with_expert.num_action_experts,
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None] * (1 + self.paligemma_with_expert.num_action_experts),
        )

        x_t = noise
        if self.dual_action_expert:
            valid_action_mask = self.manipulation_action_mask + self.mobility_action_mask
            x_t = x_t * valid_action_mask.to(dtype=x_t.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(bsize),
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
        """Apply one denoising step of the noise x_t at a given timestep.

        Args:
            state: (B, state_dim) robot state.
            prefix_pad_masks: (B, Np) prefix pad masks computed from embed_prefix.
            past_key_values: KV cache dict for the prefix (images+language).
            x_t: (B, n_action_steps, action_dim) current noisy actions.
            timestep: (B,) current time in [0, 1].

        Returns:
            v_t prediction with shape (B, n_action_steps, action_dim).
        """
        suffix_streams, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        position_ids = self._make_suffix_position_ids(prefix_pad_masks, suffix_streams)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, *suffix_streams],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, *adarms_cond],
        )
        return self._project_action_outputs(outputs_embeds[1:])
