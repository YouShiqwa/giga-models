from copy import deepcopy

from giga_models.pipelines.vla.pi0.robocasa_pi05_utils import (
    ROBOCASA_MANIPULATION_ACTION_INDICES,
    ROBOCASA_MOBILITY_ACTION_INDICES,
)

from .pi05_robocasa_set_up_cutting_station import config as single_tower_config


config = deepcopy(single_tower_config)
config['project_dir'] = './experiments/vla/pi05/robocasa_set_up_cutting_station_dual'
config['models'].update(
    dual_action_expert=True,
    manipulation_action_indices=list(ROBOCASA_MANIPULATION_ACTION_INDICES),
    mobility_action_indices=list(ROBOCASA_MOBILITY_ACTION_INDICES),
    # Equal branch weights keep the combined coefficient at 1.0 while
    # preventing the 7-D arm branch from dominating the 5-D base branch.
    manipulation_loss_weight=0.5,
    mobility_loss_weight=0.5,
)
