"""Trainer entry point for the isolated Pi0.5 dual-query variant."""

from typing import Any

from giga_models.models.vla.pi0.modeling_pi05_dual_query import PI05DualQueryPolicy

from .pi0_loss import PI0Loss
from .pi05_vggt_trainer import Pi05VGGTTrainer


class Pi05DualQueryTrainer(Pi05VGGTTrainer):
    """Reuse the validated Pi0 batch/loss and TensorBoard/W&B logging paths."""

    def get_models(self, model_config: Any) -> PI05DualQueryPolicy:
        override_names = (
            'dual_action_expert',
            'manipulation_action_indices',
            'mobility_action_indices',
            'manipulation_loss_weight',
            'mobility_loss_weight',
            'arm_num_query_tokens',
            'base_num_query_tokens',
            'query_init_std',
        )
        model_overrides = {name: model_config[name] for name in override_names if name in model_config}
        policy = PI05DualQueryPolicy.from_pretrained(model_config.pretrained, **model_overrides)
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


__all__ = ['Pi05DualQueryTrainer']
