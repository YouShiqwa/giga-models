"""Trainer entry point for the isolated Pi0.5 + VGGT geometry variant."""

import os
from typing import Any

from accelerate.tracking import WandBTracker

from giga_models.models.vla.pi0.modeling_pi05_vggt import PI05VGGTPolicy

from .pi0_loss import PI0Loss
from .pi0_trainer import Pi0Trainer


_WANDB_ENV_NAMES = {
    'entity': 'WANDB_ENTITY',
    'group': 'WANDB_RUN_GROUP',
    'job_type': 'WANDB_JOB_TYPE',
    'mode': 'WANDB_MODE',
    'name': 'WANDB_NAME',
    'notes': 'WANDB_NOTES',
}


def _tracker_enabled(log_with: str | list[str] | None, tracker_name: str) -> bool:
    if isinstance(log_with, str):
        return log_with == tracker_name
    return log_with is not None and tracker_name in log_with


def _configure_wandb_environment(wandb_config: dict[str, Any]) -> None:
    """Translate the local config into settings consumed by ``wandb.init``."""
    for config_name, env_name in _WANDB_ENV_NAMES.items():
        value = wandb_config.get(config_name)
        if value is not None:
            os.environ[env_name] = str(value)

    tags = wandb_config.get('tags')
    if tags is not None:
        if isinstance(tags, str):
            os.environ['WANDB_TAGS'] = tags
        else:
            os.environ['WANDB_TAGS'] = ','.join(str(tag) for tag in tags)


def _configure_wandb_tracker(
    log_with: str | list[str] | None,
    wandb_config: dict[str, Any],
) -> str | WandBTracker | list[str | WandBTracker] | None:
    """Use an explicit W&B project without coupling it to ``project_dir``."""
    project = wandb_config.get('project')
    if project is None:
        _configure_wandb_environment(wandb_config)
        return log_with

    init_kwargs = {
        name: wandb_config[name]
        for name in ('entity', 'group', 'job_type', 'mode', 'name', 'notes')
        if wandb_config.get(name) is not None
    }
    tags = wandb_config.get('tags')
    if tags is not None:
        init_kwargs['tags'] = [tags] if isinstance(tags, str) else list(tags)
    tracker = WandBTracker(str(project), **init_kwargs)

    if isinstance(log_with, str):
        return tracker
    return [tracker if logger == 'wandb' else logger for logger in log_with or []]


class Pi05VGGTTrainer(Pi0Trainer):
    """Reuse Pi0Trainer's batch/loss path while loading the new model class."""

    def __init__(
        self,
        *args: Any,
        wandb: dict[str, Any] | None = None,
        log_with: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.wandb_config = dict(wandb or {})
        wandb_enabled = _tracker_enabled(log_with, 'wandb')
        configured_log_with = log_with
        if wandb_enabled:
            configured_log_with = _configure_wandb_tracker(log_with, self.wandb_config)

        super().__init__(*args, log_with=configured_log_with, **kwargs)

        run_config = self.wandb_config.get('config')
        if self.is_main_process and wandb_enabled and run_config:
            self.accelerator.get_tracker('wandb').store_init_configuration(dict(run_config))

    def print_step(self) -> None:
        """Log the current LR together with the losses to every tracker."""
        if self.is_main_process and self.cur_step % self.log_interval == 0:
            learning_rate = float(self.scheduler.get_last_lr()[0])
            self._outputs['learning_rate'] = {'sum': learning_rate, 'num': 1}
        super().print_step()

    def get_models(self, model_config: Any) -> PI05VGGTPolicy:
        override_names = (
            'dual_action_expert',
            'manipulation_action_indices',
            'mobility_action_indices',
            'manipulation_loss_weight',
            'mobility_loss_weight',
            'vggt_repo_path',
            'vggt_checkpoint_path',
            'vggt_image_resolution',
            'vggt_patch_size',
            'vggt_feature_dim',
            'vggt_num_views',
            'vggt_output_grid_size',
            'vggt_norm_groups',
            'vggt_enable_alignment',
            'vggt_view_order',
        )
        model_overrides = {name: model_config[name] for name in override_names if name in model_config}
        policy = PI05VGGTPolicy.from_pretrained(model_config.pretrained, **model_overrides)

        policy.to(self.device)
        # Load before Accelerator/FSDP/torch.compile wrapping.  The external
        # frozen model is deliberately absent from registered parameters.
        policy.load_vggt(self.device)
        policy.train()

        self.loss_func = PI0Loss(
            dual_action_expert=policy.dual_action_expert,
            manipulation_action_indices=policy.manipulation_action_indices,
            mobility_action_indices=policy.mobility_action_indices,
            manipulation_loss_weight=policy.manipulation_loss_weight,
            mobility_loss_weight=policy.mobility_loss_weight,
        ).to(self.device)
        return policy


__all__ = ['Pi05VGGTTrainer']
