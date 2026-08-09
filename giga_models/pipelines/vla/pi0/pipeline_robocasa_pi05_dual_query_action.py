"""RoboCasa inference pipeline for dual-query/action-feedback Pi0.5."""

import torch

from ....models.vla.pi0.modeling_pi05_dual_query_action import PI05DualQueryActionPolicy
from ...pipeline import BasePipeline
from .pi0_utils import ImageTransform, Normalize, PadStatesAndActions, PromptTokenizerTransform, Unnormalize
from .pipeline_robocasa_pi05 import RoboCasaPi05Pipeline
from .robocasa_pi05_utils import ROBOCASA_IMAGE_KEYS, RoboCasaOutputs, load_robocasa_norm_stats


class RoboCasaPi05DualQueryActionPipeline(RoboCasaPi05Pipeline):
    """Keep the validated three-camera preprocessing while loading the new model."""

    def __init__(
        self,
        model_path: str,
        tokenizer_model_path: str,
        norm_stats_path: str,
        state_input_order: str = 'openpi',
        use_quantiles: bool = False,
        original_action_dim: int = 12,
    ) -> None:
        # RoboCasaPi05Pipeline.__init__ loads PI0Policy, so select the additive
        # model class here while keeping every preprocessing setting identical.
        BasePipeline.__init__(self)
        if state_input_order not in ('openpi', 'dataset'):
            raise ValueError("state_input_order must be either 'openpi' or 'dataset'.")

        self.policy = PI05DualQueryActionPolicy.from_pretrained(model_path)
        if (
            not self.policy.pi05_enabled
            or not self.policy.dual_action_expert
            or not self.policy.dual_query_bridge_enabled
            or not self.policy.dual_query_action_attention_enabled
        ):
            raise ValueError(
                'RoboCasaPi05DualQueryActionPipeline requires a dual-action '
                'Pi0.5 query/action-feedback checkpoint.'
            )
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


__all__ = ['RoboCasaPi05DualQueryActionPipeline']
