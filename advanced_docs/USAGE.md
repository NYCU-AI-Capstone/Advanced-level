# ShellBench 使用指南

本文件說明如何使用 ShellBench 來生成資料、訓練 policy、執行評估，以及跑完整的實驗 sweep。

---

## 前置需求

### 環境設定

ShellBench 運行在 Isaac Sim 容器中。確認你已經可以成功執行現有 task（如 CupStacking）。

```bash
# 建構 Docker image
make build

# 啟動容器（含 GPU 支援）
make launch
```

### 確認 Task 註冊成功

進入容器後，可以用以下方式快速驗證 gym 註冊：

```python
import gymnasium as gym
import simulator.tasks  # 觸發所有 task 的 gym.register()

env = gym.make("HCIS-ShellGame-SingleArm-v0")
print("ShellGame task registered successfully!")
env.close()
```

---

## Step 1: 資料生成（Demo Collection）

ShellBench 使用專用的資料生成腳本 `generate_shell_game.py`，不需要 UMI pipeline 的 `object_poses.json`。

### 基本用法

```bash
python scripts/datagen/generate_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/shell_game_3cups_2shuffles.hdf5 \
    --num_demos 100 \
    --num_cups 3 \
    --num_shuffles 2 \
    --seed 42
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num_demos` | 100 | 要生成的 demo 數量 |
| `--num_cups` | 3 | 杯子數量 (3/4/5) |
| `--num_shuffles` | 2 | 洗牌交換次數 |
| `--shuffle_speed` | 1.0 | 洗牌速度倍率 (0.5 慢 / 1.0 中 / 2.0 快) |
| `--ball_position` | random | 球的初始位置 (random 或指定 cup index) |
| `--seed` | 42 | 隨機種子，確保可復現 |
| `--dataset_file` | ./datasets/shell_game.hdf5 | 輸出檔案路徑 |

### 使用 LeRobot Recorder

如果要直接生成 LeRobot format 的資料：

```bash
python scripts/datagen/generate_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id "your_username/shell_game_demo" \
    --lerobot_dataset_fps 30 \
    --dataset_file ./datasets/shell_game.hdf5 \
    --num_demos 100 --num_cups 3 --num_shuffles 2
```

### 繼續錄製（Resume）

```bash
python scripts/datagen/generate_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --resume \
    --dataset_file ./datasets/shell_game_3cups_2shuffles.hdf5 \
    --num_demos 200  # 繼續錄到 200 筆
```

---

## Step 2: 訓練 Policy

使用 LeRobot 框架訓練 Diffusion Policy。訓練指令取決於你的 LeRobot 配置。

### 範例（Diffusion Policy）

```bash
# 將 HDF5 轉成 LeRobot 格式（如果沒用 lerobot_recorder）
# 請參考 LeRobot 文件

# 訓練
python -m lerobot.scripts.train \
    --policy.type=diffusion \
    --dataset.repo_id="your_username/shell_game_demo" \
    --output_dir=./checkpoints/shell_game_dp \
    --training.num_epochs=100
```

> **提示：** Observation horizon 是關鍵超參數。Shell game 是 non-Markovian 任務，增大 observation horizon 可能提升 DSR。建議用 exp3 實驗測試不同 horizon。

---

## Step 3: 評估（Evaluation）

### 基本用法

```bash
python scripts/eval_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --device cuda \
    --enable_cameras \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path ./checkpoints/shell_game_dp \
    --num_episodes 50 \
    --num_cups 3 \
    --num_shuffles 2 \
    --seed 42 \
    --output_json ./results/eval_3cups_2shuffles.json
```

### 參數說明

| 參數 | 說明 |
|------|------|
| `--policy_type` | LeRobot policy 類型，如 `lerobot-diffusion`、`lerobot-smolvla` |
| `--policy_checkpoint_path` | 訓練好的 checkpoint 路徑 |
| `--policy_action_horizon` | 每次 policy call 執行幾步 action (預設 16) |
| `--num_episodes` | 評估的 episode 數量 |
| `--output_json` | 結果輸出路徑 |

### 輸出格式

評估結果會存成 JSON：

```json
{
  "config": {
    "num_cups": 3,
    "num_shuffles": 2,
    "shuffle_speed": 1.0,
    "policy_checkpoint": "./checkpoints/shell_game_dp"
  },
  "num_episodes": 50,
  "DSR": 0.42,
  "MSR": 0.88,
  "SR": 0.37,
  "kappa": 0.13,
  "per_episode": [
    {"episode": 0, "selected_cup": 1, "ball_true": 2, "dsr": false, "msr": true, "sr": false},
    ...
  ]
}
```

### 指標說明

| 指標 | 公式 | 意義 |
|------|------|------|
| **DSR** | 正確選擇次數 / 總 episode 數 | 測量記憶能力 |
| **MSR** | 成功掀杯次數 / 總 episode 數 | 測量操作能力 |
| **SR** | 正確選擇且成功掀杯 / 總 episode 數 | 整體成功率 |
| **κ** | (DSR − chance) / (1 − chance) | 校正隨機猜對的 Cohen's Kappa |

其中 `chance = 1/num_cups`（3杯 = 33.3%，4杯 = 25%，5杯 = 20%）。

---

## Step 4: 跑實驗 Sweep

### 單一實驗

使用 `run_sweep.py` 自動展開所有參數組合：

```bash
# Exp 1: Shuffle 次數 vs. 成功率
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
    --output_dir results/exp1_shuffle_scaling
```

這會自動展開 `num_shuffles: [0, 1, 2, 3, 4, 5]`，依序執行：
1. 對每個 `num_shuffles` 值生成 demo 資料
2. 訓練 policy
3. 評估並存結果

### 其他實驗

```bash
# Exp 2: 杯子數量 vs. 成功率
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp2_num_cups/sweep.yaml

# Exp 3: Observation Horizon 的影響
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp3_obs_horizon/sweep.yaml

# Exp 4: Oracle Upper Bound
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp4_oracle/sweep.yaml
```

### 跳過特定步驟

```bash
# 只跑評估（已有 checkpoint）
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
    --skip_datagen --skip_training

# 只跑資料生成
python scripts/run_sweep.py \
    --base configs/base.yaml \
    --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
    --skip_training --skip_eval
```

### 結果目錄結構

```
results/exp1_shuffle_scaling/
├── summary.json                      # 所有 variant 的彙整指標
├── num_shuffles=0/
│   ├── config.yaml                   # 此 variant 的完整設定
│   ├── dataset.hdf5                  # 生成的 demo 資料
│   ├── checkpoints/                  # 訓練的 policy checkpoint
│   └── metrics.json                  # 評估結果
├── num_shuffles=1/
│   └── ...
├── num_shuffles=2/
│   └── ...
└── ...
```

---

## 快速除錯指令

### 單一設定測試（不錄製）

```bash
# 只跑環境看看動畫效果
python scripts/datagen/generate_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --num_demos 3 --num_cups 3 --num_shuffles 2
```

### 用 Override 單跑一個設定

```bash
# 測試 5 個杯子 + 5 次 shuffle 的困難設定
python scripts/datagen/generate_shell_game.py \
    --task HCIS-ShellGame-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --dataset_file ./datasets/hard_mode.hdf5 \
    --num_demos 50 --num_cups 5 --num_shuffles 5 --shuffle_speed 2.0
```

---

## 常見問題

### Q: 為什麼不修改現有的 `generate.py`？

A: `generate.py` 強耦合 UMI pipeline 的 `object_poses.json`，episode 數量由 JSON 驅動。ShellBench 的 episode 由 `--num_demos` 驅動，且 Phase 1-3 需要特殊的 hold action 處理。使用獨立腳本避免破壞已穩定的 pipeline。

### Q: Phase 1-3 期間 policy 的 observation 怎麼處理？

A: Phase 1-3 期間，`env.step(hold_action)` 照常執行，recorder 持續錄製 observation。評估時也一樣——Phase 1-3 的 observation 會被餵進 policy 的 history queue，確保 policy 有機會「看見」shuffle 過程。這是 benchmark 測試記憶能力的關鍵。

### Q: 可以同時跑多個環境 (num_envs > 1) 嗎？

A: 目前第一版只支援 `num_envs=1`。若要支援多環境平行，PhaseManager 的 truth、shuffle sequence、selected_cup 都需要改成 per-env tensor。

### Q: `any_cup_lifted` 會不會被 `rollout.py` 誤認為 success？

A: 不會。ShellBench 有專用的 `eval_shell_game.py` 來計算 DSR/MSR/SR/κ。`any_cup_lifted` 只觸發 episode 結束，不代表成功。建議不要用通用 `rollout.py` 跑 ShellBench 評估。
