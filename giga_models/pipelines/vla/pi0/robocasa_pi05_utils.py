import json
from pathlib import Path
from typing import Any

import torch


ROBOCASA_STATE_DIM = 16
ROBOCASA_ACTION_DIM = 12
ROBOCASA_MODEL_DIM = 32
ROBOCASA_MANIPULATION_ACTION_INDICES = tuple(range(0, 7))
ROBOCASA_MOBILITY_ACTION_INDICES = tuple(range(7, 12))

# Groot stores base pose first. OpenPI's RoboCasa route uses end-effector pose first.
ROBOCASA_STATE_INDICES = (7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15)
ROBOCASA_ACTION_INDICES = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)

ROBOCASA_IMAGE_KEYS = (
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
)
ROBOCASA_DATASET_IMAGE_KEY_MAP = {
    'observation.images.robot0_agentview_left': ROBOCASA_IMAGE_KEYS[0],
    'observation.images.robot0_eye_in_hand': ROBOCASA_IMAGE_KEYS[1],
    'observation.images.robot0_agentview_right': ROBOCASA_IMAGE_KEYS[2],
}
ROBOCASA_OPENPI_IMAGE_KEY_MAP = {
    'observation/image': ROBOCASA_IMAGE_KEYS[0],
    'observation/wrist_image': ROBOCASA_IMAGE_KEYS[1],
    'observation/right_image': ROBOCASA_IMAGE_KEYS[2],
}


def _reorder_last_dim(value: torch.Tensor, indices: tuple[int, ...], name: str) -> torch.Tensor:
    if value.shape[-1] != len(indices):
        raise ValueError(f'{name} must have {len(indices)} values in its last dimension, but got shape {tuple(value.shape)}.')
    index = torch.tensor(indices, dtype=torch.long, device=value.device)
    return value.index_select(-1, index).to(dtype=torch.float32)


def reorder_robocasa_state(state: torch.Tensor) -> torch.Tensor:
    """Convert the Groot dataset's 16-D state order to OpenPI RoboCasa order."""
    return _reorder_last_dim(state, ROBOCASA_STATE_INDICES, 'RoboCasa state')


def reorder_robocasa_action(action: torch.Tensor) -> torch.Tensor:
    """Convert the Groot dataset's 12-D action order to OpenPI RoboCasa order."""
    return _reorder_last_dim(action, ROBOCASA_ACTION_INDICES, 'RoboCasa action')


def map_robocasa_images(data: dict[str, Any]) -> dict[str, Any]:
    """Add canonical image keys for LeRobot, OpenPI, or GigaModels inputs."""
    output = dict(data)
    image_key_map = {**ROBOCASA_DATASET_IMAGE_KEY_MAP, **ROBOCASA_OPENPI_IMAGE_KEY_MAP}
    for source_key, target_key in image_key_map.items():
        if target_key not in output and source_key in output:
            output[target_key] = output[source_key]

    missing = [key for key in ROBOCASA_IMAGE_KEYS if key not in output]
    if missing:
        raise ValueError(f'Missing RoboCasa pi0.5 image inputs: {missing}. All three cameras are required.')
    return output


def prepare_robocasa_image(image: Any) -> torch.Tensor:
    """Convert uint8 HWC or numeric CHW/HWC images to float32 CHW in [0, 1]."""
    image = torch.as_tensor(image)
    if image.ndim != 3:
        raise ValueError(f'RoboCasa image must be 3-D, but got shape {tuple(image.shape)}.')
    if image.shape[0] == 3:
        pass
    elif image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    else:
        raise ValueError(f'RoboCasa image must have exactly three channels, but got shape {tuple(image.shape)}.')

    image = image.to(dtype=torch.float32)
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(f'RoboCasa image values must be in [0, 1] or [0, 255], but got [{minimum}, {maximum}].')
    if maximum > 1.0:
        image = image / 255.0
    return image.contiguous()


class RoboCasaDatasetInputs:
    """Adapt the local Groot-style LeRobot sample to OpenPI RoboCasa ordering."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = map_robocasa_images(data)
        for key in ROBOCASA_IMAGE_KEYS:
            output[key] = prepare_robocasa_image(output[key])
        output['observation.state'] = reorder_robocasa_state(output['observation.state'])
        if 'action' in output:
            output['action'] = reorder_robocasa_action(output['action'])
        return output


class RoboCasaOutputs:
    """Remove the model's action padding while preserving OpenPI action order."""

    def __init__(self, action_dim: int = ROBOCASA_ACTION_DIM) -> None:
        self.action_dim = action_dim

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        output['action'] = output['action'][..., : self.action_dim]
        return output


def _get_stats_entry(stats: dict[str, Any], keys: tuple[str, ...], name: str) -> dict[str, Any]:
    for key in keys:
        if key in stats:
            return stats[key]
    raise KeyError(f'Could not find {name} normalization stats. Tried keys: {keys}.')


def _prepare_stats_entry(
    stats: dict[str, Any],
    *,
    target_dim: int,
    indices: tuple[int, ...] | None,
    name: str,
) -> dict[str, list[float]]:
    padding_values = {'mean': 0.0, 'std': 1.0, 'q01': -1.0, 'q99': 1.0}
    output: dict[str, list[float]] = {}

    for field, padding_value in padding_values.items():
        if field not in stats:
            continue
        values = list(stats[field])
        if indices is not None:
            if len(values) != len(indices):
                raise ValueError(f'Raw {name} stats field {field!r} must have {len(indices)} values, but got {len(values)}.')
            values = [values[index] for index in indices]
        if len(values) > target_dim:
            raise ValueError(f'{name} stats field {field!r} has {len(values)} values, which exceeds model dimension {target_dim}.')
        output[field] = [float(value) for value in values] + [padding_value] * (target_dim - len(values))

    for required_field in ('mean', 'std'):
        if required_field not in output:
            raise KeyError(f'{name} normalization stats are missing required field {required_field!r}.')
    return output


def load_robocasa_norm_stats(path: str | Path, target_dim: int = ROBOCASA_MODEL_DIM) -> dict[str, dict[str, list[float]]]:
    """Load and pad RoboCasa stats from a raw Groot file or canonical norm file.

    A raw ``meta/stats.json`` file is reordered from Groot to OpenPI layout. A
    file wrapped in ``{"norm_stats": ...}`` is treated as already canonical and
    is only padded. Both GigaModels keys and OpenPI ``state``/``actions`` aliases
    are accepted for canonical files.
    """
    with open(path, 'r') as file:
        data = json.load(file)

    is_canonical = 'norm_stats' in data
    stats = data['norm_stats'] if is_canonical else data
    state_stats = _get_stats_entry(stats, ('observation.state', 'state'), 'state')
    action_stats = _get_stats_entry(stats, ('action', 'actions'), 'action')

    return {
        'observation.state': _prepare_stats_entry(
            state_stats,
            target_dim=target_dim,
            indices=None if is_canonical else ROBOCASA_STATE_INDICES,
            name='state',
        ),
        'action': _prepare_stats_entry(
            action_stats,
            target_dim=target_dim,
            indices=None if is_canonical else ROBOCASA_ACTION_INDICES,
            name='action',
        ),
    }
