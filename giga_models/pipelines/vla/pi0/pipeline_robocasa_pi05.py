from typing import Any

import torch

from ....models import PI0Policy
from ...pipeline import BasePipeline
from .pi0_utils import ImageTransform, Normalize, PadStatesAndActions, PromptTokenizerTransform, Unnormalize
from .robocasa_pi05_utils import (
    ROBOCASA_IMAGE_KEYS,
    ROBOCASA_STATE_DIM,
    RoboCasaOutputs,
    load_robocasa_norm_stats,
    map_robocasa_images,
    prepare_robocasa_image,
    reorder_robocasa_state,
)


class RoboCasaPi05Pipeline(BasePipeline):
    """Inference pipeline matching OpenPI's three-camera RoboCasa pi0.5 route."""

    def __init__(
        self,
        model_path: str,
        tokenizer_model_path: str,
        norm_stats_path: str,
        state_input_order: str = 'openpi',
        use_quantiles: bool = False,
        original_action_dim: int = 12,
    ) -> None:
        """Initialize a RoboCasa pi0.5 inference pipeline.

        Args:
            model_path: Converted GigaModels pi0.5 checkpoint.
            tokenizer_model_path: PaliGemma tokenizer path or hub id.
            norm_stats_path: Raw Groot ``meta/stats.json`` or canonical norm file.
            state_input_order: ``openpi`` for deployment state order or ``dataset``
                for the raw local Groot/LeRobot state order.
            use_quantiles: Whether to use q01/q99 normalization. OpenPI RoboCasa
                training uses mean/std, so the compatible default is False.
            original_action_dim: Number of RoboCasa action values to return.
        """
        super().__init__()
        if state_input_order not in ('openpi', 'dataset'):
            raise ValueError("state_input_order must be either 'openpi' or 'dataset'.")

        self.policy = PI0Policy.from_pretrained(model_path)
        if not self.policy.pi05_enabled:
            raise ValueError('RoboCasaPi05Pipeline requires a checkpoint with pi05_enabled=True.')
        self.policy.eval()

        self.device: torch.device | str = 'cpu'
        self.state_input_order = state_input_order
        self.model_dim = self.policy.max_action_dim

        norm_stats = load_robocasa_norm_stats(norm_stats_path, target_dim=self.model_dim)
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=self.model_dim)
        self.state_normalize_transform = Normalize(norm_stats['observation.state'], use_quantiles=use_quantiles)
        self.action_unnormalize_transform = Unnormalize(norm_stats['action'], use_quantiles=use_quantiles)
        self.image_transform = ImageTransform(
            resize_imgs_with_padding=(224, 224),
            present_img_keys=list(ROBOCASA_IMAGE_KEYS),
            enable_image_aug=False,
        )
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            tokenizer_model_path=tokenizer_model_path,
            max_length=200,
            discrete_state_input=True,
        )
        self.robocasa_outputs_transform = RoboCasaOutputs(action_dim=original_action_dim)

    def to(self, device: torch.device | str):
        self.device = device
        self.policy.to(device)
        self.state_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def compile(self, **kwargs) -> None:
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)

    @torch.no_grad()
    def __call__(self, images: dict[str, Any], task: str | list[str], state: Any) -> torch.Tensor:
        """Predict 50-step action chunks in OpenPI RoboCasa action order.

        ``images`` may use OpenPI, local LeRobot, or canonical GigaModels camera
        keys, with single or batched uint8 HWC / float CHW values. With the
        default ``state_input_order``, ``state`` must use
        ``[eef_pos_rel, eef_quat_rel, base_pos, base_quat, gripper_qpos]``.

        The single-sample API remains unchanged and returns ``[50, 12]``. A
        batched input returns ``[B, 50, 12]``; this only batches the existing
        preprocessing and model call and does not change model semantics.
        """
        state = torch.as_tensor(state)
        original_device = state.device
        was_single = state.ndim == 1
        if was_single:
            state = state[None, ...]
        if state.ndim != 2:
            raise ValueError(f'RoboCasa state must have shape [16] or [B, 16], but got {tuple(state.shape)}.')
        batch_size = state.shape[0]

        tasks = [task] * batch_size if isinstance(task, str) else list(task)
        if len(tasks) != batch_size or not all(isinstance(item, str) for item in tasks):
            raise ValueError(f'RoboCasa task must contain {batch_size} strings, but got {tasks!r}.')

        data = map_robocasa_images({**images, 'observation.state': state})
        state = data['observation.state'].to(device=self.device, dtype=torch.float32)
        if self.state_input_order == 'dataset':
            state = reorder_robocasa_state(state)
        elif state.shape[-1] != ROBOCASA_STATE_DIM:
            raise ValueError(
                f'OpenPI-order RoboCasa state must have {ROBOCASA_STATE_DIM} values, but got shape {tuple(state.shape)}.'
            )

        # OpenPI pads RoboCasa state before normalization/tokenization, so pi0.5
        # receives a discretized 32-D state in the prompt.
        state = self.pad_states_and_actions_transform({'observation.state': state})['observation.state']
        state = self.state_normalize_transform(state)

        batched_images = {}
        for key in ROBOCASA_IMAGE_KEYS:
            image = torch.as_tensor(data[key])
            if image.ndim == 3:
                if batch_size != 1:
                    raise ValueError(f'Unbatched image {key!r} cannot be used with batch size {batch_size}.')
                image = image[None, ...]
            if image.ndim != 4 or image.shape[0] != batch_size:
                raise ValueError(
                    f'Image {key!r} must have shape [B, H, W, 3] or [B, 3, H, W], but got {tuple(image.shape)}.'
                )
            batched_images[key] = image

        processed_image_samples: list[list[torch.Tensor]] = [[] for _ in ROBOCASA_IMAGE_KEYS]
        image_mask_samples: list[list[torch.Tensor]] = [[] for _ in ROBOCASA_IMAGE_KEYS]
        language_token_samples = []
        language_mask_samples = []
        for batch_index in range(batch_size):
            image_data = {
                key: prepare_robocasa_image(batched_images[key][batch_index]).to(self.device)
                for key in ROBOCASA_IMAGE_KEYS
            }
            sample_images, sample_masks = self.image_transform(image_data)
            for image_index, (image, mask) in enumerate(zip(sample_images, sample_masks, strict=True)):
                processed_image_samples[image_index].append(image)
                image_mask_samples[image_index].append(mask)

            language_tokens, language_masks = self.prompt_tokenizer_transform(
                {'task': tasks[batch_index], 'observation.state': state[batch_index]}
            )
            language_token_samples.append(language_tokens)
            language_mask_samples.append(language_masks)

        processed_images = [torch.stack(samples, dim=0) for samples in processed_image_samples]
        image_masks = [torch.stack(samples, dim=0) for samples in image_mask_samples]
        language_tokens = torch.stack(language_token_samples, dim=0)
        language_masks = torch.stack(language_mask_samples, dim=0)

        predicted_action = self.policy.sample_actions(
            processed_images,
            image_masks,
            language_tokens,
            language_masks,
            state=state,
        )
        predicted_action = self.action_unnormalize_transform(predicted_action)
        predicted_action = self.robocasa_outputs_transform({'action': predicted_action})['action']
        predicted_action = predicted_action.to(original_device)
        return predicted_action[0] if was_single else predicted_action
