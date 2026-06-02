#!/usr/bin/env python
"""離線診斷：用 dataset 的一整集，逐幀 recurrent 餵進訓好的 policy，
比對「預測動作 vs demo 真實動作」，看模型在 full-episode rollout 下會不會發散。

不需要 Isaac Sim。用法：
    cd /workspace/aicapstone
    python LSTM/scripts/diagnose_rollout.py \
        --ckpt LSTM/outputs/ns0_v1c/checkpoints/001000/pretrained_model \
        --repo johnnyli1220/shellbench-num_shuffles-0
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import LSTM.policy.register  # noqa: F401,E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--episode", type=int, default=0)
args = ap.parse_args()

root = f"/root/.cache/huggingface/lerobot/{args.repo}"
ds = LeRobotDataset(args.repo, root=root, video_backend="pyav")

# 取出指定 episode 的 frame index 範圍（依序）
ep_meta = ds.meta.episodes[args.episode]
frm = int(ep_meta["dataset_from_index"])
to = int(ep_meta["dataset_to_index"])
print(f"episode {args.episode}: frames [{frm}, {to}) = {to - frm} 幀")

# 載入 policy + 前後處理器
policy = get_policy_class("lstm").from_pretrained(args.ckpt, local_files_only=True)
policy.to("cuda").eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=args.ckpt,
    preprocessor_overrides={"device_processor": {"device": "cuda"},
                            "rename_observations_processor": {"rename_map": {}}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)

image_keys = sorted(policy.config.image_features.keys())
policy.reset()

per_frame_mse = []
with torch.inference_mode():
    for i in range(frm, to):
        frame = ds[i]
        obs = {k: frame[k] for k in image_keys}
        if OBS_STATE in frame:
            obs[OBS_STATE] = frame[OBS_STATE]
        proc = pre(obs)
        pred = policy.select_action(proc)        # [1, act_dim]（normalized）
        pred = post(pred).squeeze(0).cpu()        # [act_dim]（unnormalized）
        gt = frame[ACTION].cpu()                  # [act_dim]（raw）
        per_frame_mse.append(torch.mean((pred - gt) ** 2).item())

mse = torch.tensor(per_frame_mse)
n = len(mse)
print(f"\n整集 {n} 幀的「預測 vs 真實動作」MSE（unnormalized joint space）：")
bucket = max(1, n // 10)
for b in range(0, n, bucket):
    seg = mse[b:b + bucket]
    print(f"  幀 {b:4d}-{b + len(seg) - 1:4d}: mean MSE = {seg.mean().item():.5f}  max = {seg.max().item():.5f}")
print(f"\n  全集 mean MSE = {mse.mean().item():.5f}")
print(f"  前 64 幀 mean = {mse[:64].mean().item():.5f}  |  64 幀之後 mean = {mse[64:].mean().item():.5f}")
