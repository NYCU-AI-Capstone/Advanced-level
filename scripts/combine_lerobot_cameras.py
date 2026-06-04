#!/usr/bin/env python3
"""Create a LeRobot dataset with multiple camera streams combined into one image."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _repo_id_from_path(path: Path) -> str:
    name = path.name.replace("_", "-")
    return f"local/{name}"


def _to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}.")

    if image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).round().astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels after conversion, got shape {image.shape}.")

    return image


def _combine_images(images: list[np.ndarray], layout: str) -> np.ndarray:
    heights = {img.shape[0] for img in images}
    widths = {img.shape[1] for img in images}

    if layout == "horizontal":
        if len(heights) != 1:
            raise ValueError("Horizontal combine requires all images to have the same height.")
        return np.concatenate(images, axis=1)

    if layout == "vertical":
        if len(widths) != 1:
            raise ValueError("Vertical combine requires all images to have the same width.")
        return np.concatenate(images, axis=0)

    raise ValueError(f"Unsupported layout: {layout}")


def _combined_shape(image_shapes: list[tuple[int, int, int]], layout: str) -> tuple[int, int, int]:
    heights = [shape[0] for shape in image_shapes]
    widths = [shape[1] for shape in image_shapes]
    channels = {shape[2] for shape in image_shapes}

    if channels != {3}:
        raise ValueError(f"Expected all image features to have 3 channels, got {sorted(channels)}.")

    if layout == "horizontal":
        if len(set(heights)) != 1:
            raise ValueError("Horizontal combine requires all image features to have the same height.")
        return heights[0], sum(widths), 3

    if layout == "vertical":
        if len(set(widths)) != 1:
            raise ValueError("Vertical combine requires all image features to have the same width.")
        return sum(heights), widths[0], 3

    raise ValueError(f"Unsupported layout: {layout}")


def _episode_frame_range(episode: dict) -> tuple[int, int]:
    if "dataset_from_index" in episode and "dataset_to_index" in episode:
        return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if "from" in episode and "length" in episode:
        start = int(episode["from"])
        return start, start + int(episode["length"])
    if "from_index" in episode and "to_index" in episode:
        return int(episode["from_index"]), int(episode["to_index"])
    raise KeyError(f"Could not infer episode frame range from keys: {sorted(episode)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LeRobot dataset with multiple camera keys into a new dataset "
            "with one combined image key for policies such as VQ-BeT."
        )
    )
    parser.add_argument("--input-root", required=True, type=Path, help="Source LeRobot dataset root.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output LeRobot dataset root.")
    parser.add_argument("--input-repo-id", default=None, help="Source repo id. Defaults to local/<input dir>.")
    parser.add_argument("--output-repo-id", default=None, help="Output repo id. Defaults to local/<output dir>.")
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=None,
        help="Camera feature keys to combine. Defaults to all source camera keys.",
    )
    parser.add_argument(
        "--output-image-key",
        default="observation.images.combined",
        help="Name of the single combined image feature in the output dataset.",
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="How to concatenate camera images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root first if it already exists.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit for quick validation runs.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Threads used by LeRobot while writing PNG frames before video encoding.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=4,
        help="Threads used per video encoder. Lower this if CPU usage is too high.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec for the output dataset. Use h264 if AV1 encoding is too slow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    input_repo_id = args.input_repo_id or _repo_id_from_path(input_root)
    output_repo_id = args.output_repo_id or _repo_id_from_path(output_root)

    source = LeRobotDataset(input_repo_id, root=input_root)
    image_keys = args.image_keys or list(source.meta.camera_keys)
    if len(image_keys) < 2:
        raise ValueError(f"Need at least two image keys to combine, got {image_keys}.")

    missing = [key for key in image_keys if key not in source.features]
    if missing:
        raise ValueError(f"Image keys not found in source dataset: {missing}")

    image_shapes = [tuple(source.features[key]["shape"]) for key in image_keys]
    combined_shape = _combined_shape(image_shapes, args.layout)

    output_features = {
        key: value
        for key, value in source.features.items()
        if key not in source.meta.camera_keys
        and key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    }
    output_features[args.output_image_key] = {
        "dtype": "video",
        "shape": combined_shape,
        "names": ["height", "width", "channels"],
    }

    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=source.fps,
        features=output_features,
        root=output_root,
        robot_type=source.meta.robot_type,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        encoder_threads=args.encoder_threads,
        vcodec=args.vcodec,
    )

    total_episodes = source.num_episodes
    if args.max_episodes is not None:
        total_episodes = min(total_episodes, args.max_episodes)

    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Combining keys: {image_keys} -> {args.output_image_key}")
    print(f"Combined shape: {combined_shape}")
    print(f"Episodes: {total_episodes}")

    for episode_index in range(total_episodes):
        episode = source.meta.episodes[episode_index]
        start, end = _episode_frame_range(episode)

        for idx in range(start, end):
            item = source[idx]
            images = [_to_hwc_uint8(item[key]) for key in image_keys]
            combined = _combine_images(images, args.layout)

            frame = {
                key: item[key].detach().cpu().numpy() if isinstance(item[key], torch.Tensor) else item[key]
                for key in output_features
                if key != args.output_image_key
            }
            frame[args.output_image_key] = combined
            frame["task"] = item["task"]

            output.add_frame(frame)

        output.save_episode()
        print(f"Saved episode {episode_index + 1}/{total_episodes}")

    output.stop_image_writer()
    print("Done.")


if __name__ == "__main__":
    main()
