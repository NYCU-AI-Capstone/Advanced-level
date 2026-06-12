#!/usr/bin/env python
"""把「影片(video)」格式的 LeRobot dataset 在本地解碼成「圖片(image)」格式，存到本地。

目的：影片(AV1) dataset 每次讀都要 CPU 解碼 → 訓練嚴重 CPU-bound（GPU 在等）。
轉成 image 格式後，訓練直接讀圖檔、不用解碼 → GPU-bound、快很多，
才有可能在合理時間內練到 ~10 萬步（grasp 需要的量級）。

效能關鍵：**整段一次解碼**（每集每相機用一個 timestamp list 一次 decode），
而不是逐幀 src[i]（逐張隨機 seek 在 AV1 上慢 ~9×）；再加 **平行寫 PNG**。

只在「本地」產生 image 版（不 push HF），訓練時用 --dataset.root 指過去即可。

⚠️ --dst-root 要放在「持久化、會被掛載到 host」的路徑（如 repo 的 data/ 底下），
不要放 /root/.cache —— 容器是 --rm，cache 重開就沒了。

用法（容器內）：
    cd /workspace/aicapstone
    python policies/lstm/scripts/decode_dataset_to_images.py \
        --src-repo johnnyli1220/shellbench-num_shuffles-3 \
        --src-root /root/.cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-3 \
        --dst-root /workspace/aicapstone/data/lerobot_img/johnnyli1220/shellbench-num_shuffles-3
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import policies.lstm.scripts.setup_cache  # noqa: F401,E402

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import decode_video_frames

ap = argparse.ArgumentParser()
ap.add_argument("--src-repo", required=True)
ap.add_argument("--src-root", required=True)
ap.add_argument("--dst-root", required=True)
ap.add_argument("--max-episodes", type=int, default=None, help="只轉前 N 集（測試用）")
ap.add_argument("--writer-threads", type=int, default=8, help="平行寫 PNG 的執行緒數")
ap.add_argument(
    "--resize",
    type=int,
    default=0,
    help="把相機影像縮成 NxN 再存（0=不縮，存原始解析度）。"
    "強烈建議設成訓練用的 image_size（如 128）—— 否則全解析度 + 長 seq_len 會讓 dataloader 吃爆 RAM(OOM)。",
)
args = ap.parse_args()

src = LeRobotDataset(args.src_repo, root=args.src_root, video_backend="pyav")
print(f"source: {src.num_episodes} episodes, {src.num_frames} frames, fps={src.fps}")
tolerance_s = getattr(src, "tolerance_s", 1.0 / src.fps)

# 影像 feature 的 dtype 從 "video" 改成 "image"（create(use_videos=False) 不接受 video dtype）
dst_features = {}
for key, ft in src.meta.features.items():
    ft = dict(ft)
    if ft.get("dtype") == "video":
        ft["dtype"] = "image"
    if args.resize > 0 and key.startswith("observation.images."):
        ft["shape"] = [args.resize, args.resize, 3]  # 縮放後的 HWC 形狀
    dst_features[key] = ft

dst_root = pathlib.Path(args.dst_root)
if dst_root.exists():
    raise SystemExit(f"目標已存在，請先刪除或換路徑：{dst_root}")

dst = LeRobotDataset.create(
    repo_id=args.src_repo,
    fps=src.fps,
    features=dst_features,
    root=str(dst_root),
    robot_type=src.meta.robot_type,
    use_videos=False,
)
# WORKAROUND: lerobot's save_episode() calls clear_episode_buffer(delete_images=True)
# which deletes the images we just wrote (designed for video encoding workflow).
# Patch it to never delete images since they ARE the final output.
_original_clear = dst.clear_episode_buffer.__func__
def _clear_keep_images(self, delete_images=True):
    _original_clear(self, delete_images=False)
import types
dst.clear_episode_buffer = types.MethodType(_clear_keep_images, dst)

image_keys = [k for k in src.meta.features if k.startswith("observation.images.")]
data_keys = [k for k in src.meta.features if k not in DEFAULT_FEATURES and k not in image_keys]

# task_index -> task string（add_frame 需要 task 字串）
tasks_df = src.meta.tasks
idx_to_task = {int(tasks_df.loc[t, "task_index"]): t for t in tasks_df.index}

n_eps = src.num_episodes if args.max_episodes is None else min(args.max_episodes, src.num_episodes)
for ep in range(n_eps):
    em = src.meta.episodes[ep]
    frm, to = int(em["dataset_from_index"]), int(em["dataset_to_index"])
    length = to - frm

    # 1) 整段一次解碼每個相機（快）
    cam_frames = {}
    for cam in image_keys:
        vpath = src.root / src.meta.get_video_file_path(ep, cam)
        from_ts = float(em[f"videos/{cam}/from_timestamp"])
        ts = [from_ts + i / src.fps for i in range(length)]
        frames = decode_video_frames(str(vpath), ts, tolerance_s, backend="pyav")  # [L, C, H, W]
        if args.resize > 0:
            frames = F.interpolate(
                frames, size=(args.resize, args.resize), mode="bilinear", align_corners=False
            )
        cam_frames[cam] = frames

    # 2) 逐幀組裝（影像用上面解好的；action/state 直接讀 parquet、不解碼）
    for j in range(length):
        row = src.hf_dataset[frm + j]
        frame = {}
        for cam in image_keys:
            img = cam_frames[cam][j]
            arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
            frame[cam] = arr
        for k in data_keys:
            frame[k] = row[k]
        frame["task"] = idx_to_task[int(row["task_index"])]
        dst.add_frame(frame)

    dst.save_episode()
    print(f"  episode {ep + 1}/{n_eps} done ({length} frames)", flush=True)

dst.finalize()  # 必要：把緩衝的 episode metadata 寫進 parquet，否則 dataset 無效

print(f"\n✅ 轉檔完成 → {dst_root}")
print("訓練時用： --dataset.repo_id=%s --dataset.root=%s" % (args.src_repo, dst_root))
