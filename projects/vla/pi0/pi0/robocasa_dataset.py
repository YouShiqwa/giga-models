from pathlib import Path

import av
import torch
from giga_datasets import register_dataset
from giga_datasets.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import lerobot_dataset as lerobot_dataset_module

from giga_models.pipelines.vla.pi0.robocasa_pi05_utils import ROBOCASA_DATASET_IMAGE_KEY_MAP


_ORIGINAL_DECODE_VIDEO_FRAMES = lerobot_dataset_module.decode_video_frames


def _decode_video_frames_native_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """Decode timestamped frames with native PyAV instead of torchvision.VideoReader."""
    if backend != 'pyav':
        return _ORIGINAL_DECODE_VIDEO_FRAMES(video_path, timestamps, tolerance_s, backend)
    if not timestamps:
        raise ValueError('At least one video timestamp is required.')

    query_timestamps = torch.tensor(timestamps, dtype=torch.float64)
    first_timestamp = min(timestamps)
    last_timestamp = max(timestamps)
    loaded_frames: list[torch.Tensor] = []
    loaded_timestamps: list[float] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise RuntimeError(f'Video stream has no time base: {video_path}')
        stream.thread_type = 'AUTO'

        seek_offset = max(0, int(first_timestamp / float(stream.time_base)))
        container.seek(seek_offset, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            frame_array = frame.to_ndarray(format='rgb24')
            loaded_frames.append(torch.from_numpy(frame_array).permute(2, 0, 1))
            loaded_timestamps.append(timestamp)
            if timestamp >= last_timestamp:
                break

    if not loaded_frames:
        raise RuntimeError(f'No video frames could be decoded from {video_path}.')

    decoded_timestamps = torch.tensor(loaded_timestamps, dtype=torch.float64)
    distances = torch.cdist(query_timestamps[:, None], decoded_timestamps[:, None], p=1)
    minimum_distances, closest_indices = distances.min(dim=1)
    if not bool((minimum_distances < tolerance_s).all()):
        raise RuntimeError(
            f'Closest decoded frame exceeds tolerance {tolerance_s} for {video_path}: '
            f'query={timestamps}, minimum_distance={minimum_distances.tolist()}.'
        )

    closest_frames = torch.stack([loaded_frames[int(index)] for index in closest_indices])
    return closest_frames.to(dtype=torch.float32) / 255.0


def install_native_pyav_decoder() -> None:
    """Install the decoder only for the LeRobot module used by this data route."""
    if lerobot_dataset_module.decode_video_frames is not _decode_video_frames_native_pyav:
        lerobot_dataset_module.decode_video_frames = _decode_video_frames_native_pyav


@register_dataset
class RoboCasaLeRobotDataset(LeRobotDataset):
    """LeRobot wrapper with a torchvision-0.28-compatible PyAV decoder."""

    def __init__(self, *args, fail_on_zero_video: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on_zero_video = fail_on_zero_video

    def open(self) -> None:
        install_native_pyav_decoder()
        super().open()

    def _get_data(self, index: int) -> dict:
        data = super()._get_data(index)
        if self.fail_on_zero_video:
            camera_keys = tuple(ROBOCASA_DATASET_IMAGE_KEY_MAP)
            if all(key in data for key in camera_keys) and all(torch.count_nonzero(data[key]) == 0 for key in camera_keys):
                raise RuntimeError(
                    f'All three RoboCasa images are zero at dataset index {index}; refusing to train on a video-decoder fallback sample.'
                )
        return data
