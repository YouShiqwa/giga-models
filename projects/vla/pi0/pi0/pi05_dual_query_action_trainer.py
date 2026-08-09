"""Trainer for Pi0.5 dual queries with branch-local action feedback."""

from typing import Any

from giga_models.models.vla.pi0.modeling_pi05_dual_query_action import PI05DualQueryActionPolicy

from .pi0_loss import PI0Loss
from .pi05_vggt_trainer import Pi05VGGTTrainer


class Pi05DualQueryActionTrainer(Pi05VGGTTrainer):
    """Reuse the validated RoboCasa batch, dual loss, TensorBoard, and W&B paths."""

    def get_models(self, model_config: Any) -> PI05DualQueryActionPolicy:
        override_names = (
            'dual_action_expert',
            'manipulation_action_indices',
            'mobility_action_indices',
            'manipulation_loss_weight',
            'mobility_loss_weight',
            'arm_num_query_tokens',
            'base_num_query_tokens',
            'query_init_std',
            'action_expert_initialization',
            'action_expert_random_seed',
        )
        model_overrides = {name: model_config[name] for name in override_names if name in model_config}
        policy = PI05DualQueryActionPolicy.from_pretrained(
            model_config.pretrained,
            reset_action_experts_after_load=bool(model_config.get('reset_action_experts_after_load', False)),
            **model_overrides,
        )
        policy.to(self.device)
        policy.train()

        self.loss_func = PI0Loss(
            dual_action_expert=policy.dual_action_expert,
            manipulation_action_indices=policy.manipulation_action_indices,
            mobility_action_indices=policy.mobility_action_indices,
            manipulation_loss_weight=policy.manipulation_loss_weight,
            mobility_loss_weight=policy.mobility_loss_weight,
        ).to(self.device)
        return policy


__all__ = ['Pi05DualQueryActionTrainer']
