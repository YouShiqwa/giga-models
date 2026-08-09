from typing import Any

import torch
from giga_train import Trainer

from giga_models import PI0Policy
from .pi0_loss import PI0Loss


class Pi0Trainer(Trainer):
    def get_models(self, model_config: Any) -> PI0Policy:
        """Instantiate and prepare the PI0 model for training.

        Args:
            model_config: Config object containing `pretrained` path or hub id.

        Returns:
            A `PI0Policy` moved to the correct device and set to train mode.
        """
        override_names = (
            'dual_action_expert',
            'manipulation_action_indices',
            'mobility_action_indices',
            'manipulation_loss_weight',
            'mobility_loss_weight',
        )
        model_overrides = {name: model_config[name] for name in override_names if name in model_config}
        p0 = PI0Policy.from_pretrained(model_config.pretrained, **model_overrides)

        p0.to(self.device)
        p0.train()

        self.loss_func = PI0Loss(
            dual_action_expert=p0.dual_action_expert,
            manipulation_action_indices=p0.manipulation_action_indices,
            mobility_action_indices=p0.mobility_action_indices,
            manipulation_loss_weight=p0.manipulation_loss_weight,
            mobility_loss_weight=p0.mobility_loss_weight,
        ).to(self.device)

        return p0

    def forward_step(self, batch_dict: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
        """Perform one training step and return the loss tensor.

        Args:
            batch_dict: Preprocessed batch containing images, masks, tokens, state and actions.

        Returns:
            Loss tensor (e.g., per-sample or aggregated depending on the Trainer reduction).
        """
        images = batch_dict['images']
        img_masks = batch_dict['image_masks']
        lang_tokens = batch_dict['lang_tokens']
        lang_masks = batch_dict['lang_masks']

        state = batch_dict['observation.state']
        actions = batch_dict['action']
        action_loss_mask = batch_dict['action_loss_mask']

        noisy_model_input, timesteps = self.loss_func.add_noise(actions)
        model_pred = self.model(images, img_masks, lang_tokens, lang_masks, state, noisy_model_input, timesteps)

        loss = self.loss_func(model_pred, loss_mask=action_loss_mask)
        return loss
