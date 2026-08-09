from typing import Any

from giga_train import TRANSFORMS

from giga_models.pipelines.vla.pi0.pi0_utils import ImageTransform, Normalize, PadStatesAndActions, PromptTokenizerTransform
from giga_models.pipelines.vla.pi0.robocasa_pi05_utils import RoboCasaDatasetInputs, load_robocasa_norm_stats


@TRANSFORMS.register
class RoboCasaPi05Transform:
    """Prepare local Groot-style RoboCasa samples for pi0.5 fine-tuning."""

    def __init__(
        self,
        norm_stats_path: str,
        use_quantiles: bool = False,
        model_dim: int = 32,
        image_cfg: dict[str, Any] | None = None,
        prompt_cfg: dict[str, Any] | None = None,
    ) -> None:
        if image_cfg is None:
            raise ValueError('image_cfg is required.')
        if prompt_cfg is None:
            raise ValueError('prompt_cfg is required.')

        norm_stats = load_robocasa_norm_stats(norm_stats_path, target_dim=model_dim)
        self.robocasa_inputs_transform = RoboCasaDatasetInputs()
        self.pad_states_and_actions_transform = PadStatesAndActions(action_dim=model_dim)
        self.state_normalize_transform = Normalize(norm_stats['observation.state'], use_quantiles=use_quantiles)
        self.action_normalize_transform = Normalize(norm_stats['action'], use_quantiles=use_quantiles)
        self.image_transform = ImageTransform(**image_cfg)
        self.prompt_tokenizer_transform = PromptTokenizerTransform(**prompt_cfg)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data = self.robocasa_inputs_transform(data)

        # OpenPI's RoboCasa adapter pads before normalization and pi0.5 state
        # tokenization. The padded tail stays zero through mean/std normalization.
        data = self.pad_states_and_actions_transform(data)
        data['observation.state'] = self.state_normalize_transform(data['observation.state'])
        data['action'] = self.action_normalize_transform(data['action'])

        language_tokens, language_masks = self.prompt_tokenizer_transform(data)
        images, image_masks = self.image_transform(data)
        return {
            'images': images,
            'image_masks': image_masks,
            'lang_tokens': language_tokens,
            'lang_masks': language_masks,
            'observation.state': data['observation.state'],
            'action': data['action'],
            'action_loss_mask': ~data['action_is_pad'],
        }
