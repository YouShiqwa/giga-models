import dataclasses
import json
import pathlib
from typing import Literal

import numpy as np
import tyro
from giga_datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from giga_models.pipelines.vla.pi0.pi0_utils import AlohaInputs, DeltaActions, PadStatesAndActions
from giga_models.pipelines.vla.pi0.robocasa_pi05_utils import reorder_robocasa_action, reorder_robocasa_state


@dataclasses.dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st quantile
    q99: np.ndarray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vector_length)]
        else:
            if vector_length != self._mean.size:
                raise ValueError('The length of new vectors does not match the initialized vector length.')
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError('Cannot compute statistics for less than 2 vectors.')

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class TransformDataset(Dataset):
    def __init__(self, dataset, data_transforms, return_keys):
        self.dataset = dataset
        self.data_transforms = data_transforms
        self.return_keys = return_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        for transform in self.data_transforms:
            data = transform(data)

        result = {}
        for key in self.return_keys:
            values = np.asarray(data[key], dtype=np.float64)
            result[key] = values.reshape(-1, values.shape[-1])
        return result


class RoboCasaStatsInputs:
    """Reorder Groot state/action without requiring or decoding camera inputs."""

    def __call__(self, data):
        output = dict(data)
        output['observation.state'] = reorder_robocasa_state(output['observation.state'])
        output['action'] = reorder_robocasa_action(output['action'])
        return output


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    serializable = {
        key: {
            field.name: value.tolist()
            for field in dataclasses.fields(stats)
            if (value := getattr(stats, field.name)) is not None
        }
        for key, stats in norm_stats.items()
    }
    return json.dumps({'norm_stats': serializable}, indent=2)


def pad_norm_stats(stats: NormStats, target_dim: int) -> NormStats:
    """Pad canonical statistics using neutral normalization values."""
    current_dim = stats.mean.shape[-1]
    if current_dim > target_dim:
        raise ValueError(f'Stats dimension {current_dim} exceeds target dimension {target_dim}.')
    padding = target_dim - current_dim
    if padding == 0:
        return stats

    q01 = None if stats.q01 is None else np.pad(stats.q01, (0, padding), constant_values=-1.0)
    q99 = None if stats.q99 is None else np.pad(stats.q99, (0, padding), constant_values=1.0)
    return NormStats(
        mean=np.pad(stats.mean, (0, padding), constant_values=0.0),
        std=np.pad(stats.std, (0, padding), constant_values=1.0),
        q01=q01,
        q99=q99,
    )


def compute_norm_stats(
    data_paths: list[str],
    output_path: str,
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    action_dim: int = 32,
    adapt_to_pi: bool = True,
    dataset_type: Literal['aloha', 'robocasa'] = 'aloha',
    num_workers: int = 64,
):
    """Compute normalization statistics from multiple datasets.

    Args:
        data_paths: List of paths to the datasets.
        output_path: Path to save the computed normalization statistics.
        sample_rate: Fraction of data to use for computing statistics (0.0 to 1.0).
        action_chunk: Number of action chunks for delta computation.
        action_dim: Dimension of the action space.
        adapt_to_pi: Whether to adapt the data to PI format.
        dataset_type: Input schema. ``robocasa`` applies Groot-to-OpenPI state
            and action reordering without ALOHA or delta-action transforms.
        num_workers: Number of DataLoader workers.
    """
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(f'sample_rate must be in (0, 1], but got {sample_rate}.')
    if num_workers < 0:
        raise ValueError(f'num_workers must be non-negative, but got {num_workers}.')

    output_path = pathlib.Path(output_path)

    keys = ['observation.state', 'action']
    stats = {key: RunningStats() for key in keys}

    if dataset_type == 'robocasa':
        data_transforms = [RoboCasaStatsInputs()]
    else:
        data_transforms = [
            AlohaInputs(adapt_to_pi=adapt_to_pi),
            DeltaActions(),
            PadStatesAndActions(action_dim=action_dim),
        ]

    # Process each dataset path
    data_or_config = [
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info=dict(
                action=action_chunk,
            ),
            meta_name='meta',
            skip_video_decoding=True,
        )
        for data_path in data_paths
    ]
    dataset = load_dataset(data_or_config)

    num_frames = int(sample_rate * len(dataset))
    if num_frames < 2:
        raise ValueError(f'sample_rate selects only {num_frames} frames; at least 2 are required.')
    shuffle = False
    if sample_rate < 1.0:
        shuffle = True

    transform_dataset = TransformDataset(dataset, data_transforms, keys)
    dataloader = DataLoader(
        transform_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    # Update statistics from all datasets
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=num_frames, desc='Computing norm stats'):
        if batch_idx >= num_frames:
            break
        for key in keys:
            stats[key].update(batch_data[key][0].numpy())

    # Compute final statistics from all datasets
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    if dataset_type == 'robocasa':
        norm_stats = {key: pad_norm_stats(value, action_dim) for key, value in norm_stats.items()}

    print(f'Writing {dataset_type} stats to: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_json(norm_stats))


if __name__ == '__main__':
    tyro.cli(compute_norm_stats)
