# ShellBench Usage Guide

ShellBench 是 shell game benchmark：robot 先看球在哪個杯子旁邊，杯子蓋住球後進行 shuffle，最後 policy 只能靠記憶選杯並把杯子掀起來。

這份文件說明目前 codebase 的實際用法：如何產生資料、訓練 LeRobot policy、執行 evaluation，以及用 `run_sweep.py` 跑可重複的 benchmark 實驗。

---

## 0. 目前設計重點與限制

先記住幾個會影響使用方式的點：

| 項目 | 目前狀態 |
|------|----------|
| Task id | `HCIS-ShellGame-SingleArm-v0` |
| 支援環境數 | 只支援 `num_envs=1` |
| 支援杯數 | `num_cups` 支援 3 到 5 |
| Phase | Reveal -> Cover -> Shuffle -> Act |
| 成功判定 | 不使用 env 內建 `success` 作為最終成功；由 `eval_shell_game.py` 統一算 DSR/MSR/SR/kappa |
| 選杯判定 | Act phase 中第一次偵測到任一杯子被 lift 超過門檻時，記錄 `selected_cup_index` |
| Policy memory | Evaluation 期間 Phase 1-3 的 observation 也會餵進 LeRobot policy history，Act phase 才真正執行 policy action |
| Shuffle truth | `_ball_cup_idx` 代表實體藏球杯 id；shuffle 時 truth id 不交換，球位置跟著藏球杯同步 |
| Kinematic cups | Scripted phases 用 kinematic 控制杯子；Act phase 會切回 dynamic，讓 robot/policy 可以實際掀杯 |

ShellBench 第一版不是 parallel env benchmark。若傳入 `--num_envs > 1`，資料生成腳本會直接報錯。

---

## 1. 環境準備

### 1.1 Simulator / Datagen / Evaluation

資料生成和模擬評估需要 Isaac Sim / Isaac Lab 環境。通常在專案容器內執行：

```bash
make build
make launch
```

進入容器後，確認 task 可以註冊：

```bash
python - <<'PY'
import gymnasium as gym
import simulator.tasks  # triggers task registration

env = gym.make("HCIS-ShellGame-SingleArm-v0")
print("ShellGame task registered successfully")
env.close()
PY
```

### 1.2 LeRobot Training

LeRobot training 建議在 host machine 的 `uv` 環境跑，避免 Docker I/O 和 GPU passthrough 造成訓練變慢。詳細背景可看 `docs/lerobot_training.md`。

```bash
uv sync
source .venv/bin/activate
export HF_USER=<your-huggingface-username>
```

確認可以呼叫：

```bash
lerobot-train --help
```

如果要 push/pull Hugging Face dataset 或 policy，需要先登入：

```bash
hf auth login
```

---

## 2. 手動產生資料

ShellBench 使用專用腳本 `scripts/datagen/generate_shell_game.py`，不走 UMI pipeline 的 `object_poses.json`。Episode 數量由 `--num_demos` 控制，每個 episode 會依 seed 隨機決定球位置與 shuffle sequence。

### 2.1 快速產生 HDF5 demo

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
  --shuffle_speed 1.0 \
  --ball_position random \
  --seed 42
```

### 2.2 直接錄成 LeRobot dataset

若後續要用 LeRobot training，建議直接啟用 LeRobot recorder：

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --use_lerobot_recorder \
  --lerobot_dataset_repo_id "${HF_USER}/shellbench-demo" \
  --lerobot_dataset_fps 30 \
  --dataset_file ./datasets/shell_game.hdf5 \
  --num_demos 100 \
  --num_cups 3 \
  --num_shuffles 2 \
  --seed 42
```

`--lerobot_dataset_repo_id` 是 LeRobot dataset 的 Hugging Face repo id，例如 `your_name/shellbench-demo`。`--dataset_file` 仍會被 recorder config 使用，作為這次輸出的本地檔名/位置。

### 2.3 Phase duration 參數

這些參數會同時影響資料生成和評估，建議 training/eval 保持一致：

| 參數 | 預設 | 說明 |
|------|------|------|
| `--reveal_frames` | 50 | 球可見、杯子尚未遮住前的觀察長度 |
| `--cover_frames` | 10 | 杯子蓋住球的過渡長度 |
| `--shuffle_per_swap_frames` | 30 | 每次 swap 的動畫長度 |
| `--act_frames` | 150 | Act phase 最長步數 |

範例：

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/slow_shuffle.hdf5 \
  --num_demos 50 \
  --num_cups 5 \
  --num_shuffles 4 \
  --reveal_frames 80 \
  --shuffle_per_swap_frames 45 \
  --act_frames 180
```

### 2.4 Resume 錄製

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --resume \
  --dataset_file ./datasets/shell_game_3cups_2shuffles.hdf5 \
  --num_demos 200
```

`--num_demos 200` 代表這次目標總數是 200，不是額外新增 200。

---

## 3. 手動訓練 Policy

LeRobot training 的核心指令如下：

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/shellbench-demo \
  --policy.type=diffusion \
  --output_dir=results/manual_shellbench/checkpoints \
  --job_name=shellbench_manual \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/shellbench-policy-manual \
  --policy.n_obs_steps=8 \
  --training.num_epochs=100
```

重要參數：

| 參數 | 說明 |
|------|------|
| `--dataset.repo_id` | 要訓練的 LeRobot dataset repo id |
| `--dataset.root` | 可選；若資料已經在本地，可指定本地 dataset 目錄避免重新下載 |
| `--policy.type` | Policy 類型，例如 `diffusion` 或其他 LeRobot 支援的 policy |
| `--policy.n_obs_steps` | Observation horizon。Shell game 需要記憶，這是關鍵超參數 |
| `--output_dir` | 訓練輸出資料夾；完成後通常會有 `pretrained_model/` |
| `--policy.repo_id` | 訓練完成後 policy 要 push 到的 Hugging Face repo id |
| `--training.num_epochs` | 訓練 epoch 數 |
| `--training.batch_size` | Batch size；OOM 時可以調小 |

若你已經有本地 LeRobot dataset，例如 `.cache/lerobot/${HF_USER}/shellbench-demo`：

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/shellbench-demo \
  --dataset.root=.cache/lerobot/${HF_USER}/shellbench-demo \
  --policy.type=diffusion \
  --output_dir=results/manual_shellbench/checkpoints \
  --job_name=shellbench_manual \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.n_obs_steps=8 \
  --training.num_epochs=100
```

---

## 4. 手動評估

評估腳本是 `scripts/eval_shell_game.py`。它會跑完整 Reveal/Cover/Shuffle/Act episode，並輸出 DSR、MSR、SR、kappa。

### 4.1 評估 LeRobot policy

```bash
python scripts/eval_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda \
  --enable_cameras \
  --policy_backend lerobot \
  --policy_type lerobot-diffusion \
  --policy_checkpoint_path ./results/manual_shellbench/checkpoints/pretrained_model \
  --policy_action_horizon 16 \
  --num_episodes 50 \
  --num_cups 3 \
  --num_shuffles 2 \
  --shuffle_speed 1.0 \
  --seed 42 \
  --output_json ./results/manual_shellbench/metrics.json
```

Phase 1-3 期間 evaluator 會呼叫 `policy.observe(...)` 把 observation 餵進 policy history，但會丟掉這些 scripted phase 產生的 action queue。Act phase 開始後才使用 policy action 控制 robot。

### 4.2 評估 oracle upper bound

Oracle 使用 scripted FSM 直接選 ground-truth cup，主要用來檢查 task / manipulation 上限：

```bash
python scripts/eval_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda \
  --enable_cameras \
  --policy_backend oracle \
  --num_episodes 50 \
  --num_cups 3 \
  --num_shuffles 2 \
  --seed 42 \
  --output_json ./results/oracle/metrics.json
```

Oracle 不需要 `--policy_checkpoint_path`。

### 4.3 Evaluation metrics

輸出 JSON 大致如下：

```json
{
  "config": {
    "num_cups": 3,
    "num_shuffles": 2,
    "shuffle_speed": 1.0,
    "policy_backend": "lerobot",
    "policy_checkpoint": "./results/manual_shellbench/checkpoints/pretrained_model"
  },
  "num_episodes": 50,
  "DSR": 0.42,
  "MSR": 0.88,
  "SR": 0.37,
  "kappa": 0.13,
  "per_episode": [
    {
      "episode": 0,
      "selected_cup": 1,
      "ball_true": 2,
      "dsr": false,
      "msr": true,
      "sr": false
    }
  ]
}
```

| 指標 | 意義 |
|------|------|
| DSR | Decision Success Rate：選到正確藏球杯的比例 |
| MSR | Manipulation Success Rate：有成功掀起任一杯的比例 |
| SR | Success Rate：同時選對且完成掀杯的比例 |
| kappa | Chance-corrected DSR，`(DSR - 1/num_cups) / (1 - 1/num_cups)` |

注意：掀起杯子不等於成功。`any_cup_lifted` 只代表動作完成，真正成功與否由 evaluator 在 episode 結束後判斷。

---

## 5. 用 Sweep 跑 Benchmark

`scripts/run_sweep.py` 會讀取一份 base config 和一份 sweep config，展開所有參數組合，對每個 variant 依序執行：

1. Datagen
2. Training
3. Evaluation
4. 寫出每個 variant 的 `metrics.json` 和整體 `summary.json`

### 5.1 基本指令

```bash
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --output_dir results/exp1_shuffle_scaling
```

如果沒有指定 `--output_dir`，會使用 `results/<experiment_name>`。

### 5.2 現有實驗

```bash
# Exp 1: 固定 3 cups，掃 num_shuffles = 0..5
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml

# Exp 2: 固定 2 shuffles，掃 num_cups = 3,4,5
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp2_num_cups/sweep.yaml

# Exp 3: 固定任務難度，掃 policy.observation_horizon = 2,8,16,32
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp3_obs_horizon/sweep.yaml

# Exp 4: Oracle upper bound，不訓練
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp4_oracle/sweep.yaml
```

### 5.3 跳過步驟

```bash
# 只用既有 checkpoint 跑 evaluation
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --skip_datagen \
  --skip_training

# 只生成 dataset，不訓練、不評估
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --skip_training \
  --skip_eval
```

---

## 6. Sweep Config 說明

`configs/base.yaml` 是預設設定，`configs/experiments/*/sweep.yaml` 只列出要掃的欄位。Sweep key 使用 dotted path，例如 `task.num_shuffles` 或 `policy.observation_horizon`。

### 6.1 Task 與 phase

```yaml
task:
  num_cups: 3
  num_shuffles: 2
  shuffle_speed: 1.0
  ball_position: random

phase_duration:
  reveal: 50
  cover: 10
  shuffle_per_swap: 30
  act: 150
```

這些值會同時傳給 datagen 和 eval。Benchmark 中同一個 variant 的 training/eval 應該使用一致的 phase 設定。

### 6.2 Policy 與 training backend

```yaml
policy:
  backend: lerobot        # eval backend: lerobot / oracle
  type: diffusion
  observation_horizon: 2
  action_horizon: 16
  checkpoint: null
  repo_id: null
  train_args: {}
  eval_args: []

training:
  backend: lerobot        # lerobot / command / pretrained / none
  executable: lerobot-train
  dataset_repo_prefix: ${HF_USER}/shellbench
  policy_repo_prefix: ${HF_USER}/shellbench-policy
  device: cuda
  wandb_enable: true
  job_name_prefix: shellbench
  output_dir: "{checkpoint_dir}"
  checkpoint_path: "{checkpoint_dir}/pretrained_model"
  extra_args:
    - "--training.num_epochs=3"
    - "--training.batch_size=8"
  command: null
```

Backend 用法：

| 設定 | 用途 |
|------|------|
| `policy.backend: lerobot` + `training.backend: lerobot` | 預設路徑：生成資料、跑 `lerobot-train`、再評估 checkpoint |
| `policy.backend: oracle` + `training.backend: none` | Oracle upper bound，不需要 dataset 或 checkpoint |
| `policy.backend: lerobot` + `training.backend: pretrained` | 不訓練，使用 `policy.checkpoint` 指向既有 LeRobot checkpoint |
| `training.backend: command` | 用 `training.command` 自訂訓練命令，適合接其他方法 |
| `training.backend: none` | 完全跳過 training，通常搭配 oracle 或外部流程 |

目前 `scripts/eval_shell_game.py` 的 `--policy_backend` 只接受 `lerobot` 和 `oracle`。若要評估既有 checkpoint，請保持 `policy.backend: lerobot`，並把 `training.backend` 設成 `pretrained`。

`policy.observation_horizon` 會自動映射成 LeRobot training 的 `--policy.n_obs_steps=<value>`，除非你已經在 `policy.train_args` 裡明確設定 `policy.n_obs_steps`。

`training.extra_args` 是 list，每一項會原樣附加到 `lerobot-train` 後面。例如：

```yaml
training:
  extra_args:
    - "--training.num_epochs=100"
    - "--training.batch_size=32"
    - "--optimizer.lr=1e-4"
```

### 6.3 Dataset repo 與 local reuse

```yaml
data_collection:
  method: fsm_planner
  num_demos: 100
  use_lerobot_recorder: true
  lerobot_dataset_fps: 30
  dataset_repo_id: null

  # 1. If this local path exists and local_reuse_if_exists=true,
  #    skip datagen and train with --dataset.root=<local_dataset_dir>.
  # 2. Otherwise, if HF_reuse_if_exists=true and dataset_repo_id exists on
  #    Hugging Face, skip datagen and let lerobot-train load/cache by repo id.
  # 3. Otherwise, generate a new dataset for this variant.
  local_dataset_dir: ".cache/lerobot/{dataset_repo_id}"
  local_reuse_if_exists: true
  HF_reuse_if_exists: true
```

Repo id 生成規則：

| 欄位 | 若為 `null` 時的預設 |
|------|----------------------|
| `data_collection.dataset_repo_id` | `{training.dataset_repo_prefix}-{variant_name}` |
| `policy.repo_id` | `{training.policy_repo_prefix}-{variant_name}` |

例如 `HF_USER=alice`、variant 是 `num_shuffles=2`：

```text
dataset_repo_id = alice/shellbench-num_shuffles-2
policy_repo_id  = alice/shellbench-policy-num_shuffles-2
```

Reuse priority：

1. 若 `.cache/lerobot/{dataset_repo_id}` 已存在，`run_sweep.py` 會跳過 datagen，training 時加上 `--dataset.root=<local_dataset_dir>`。
2. 若本地不存在，但 Hugging Face 上已經有 `dataset_repo_id`，會跳過 datagen，training 仍透過 `--dataset.repo_id=<dataset_repo_id>` 讓 LeRobot 自己解析或下載。
3. 若本地和 HF 都沒有，才會執行 Isaac datagen。

目前要特別注意：`HF_reuse_if_exists` 只做「檢查 HF 後跳過 datagen」，不會主動把 HF dataset 下載到專案 `.cache/`。因此如果你要求所有 cache 都必須落在專案目錄，還需要在 runner 補上 explicit `huggingface_hub.snapshot_download(..., local_dir=.cache/...)` 流程。

---

## 7. 接不同方法作為 Benchmark

若要比較 LeRobot 以外的方法，建議把 ShellBench 當成固定 datagen/eval harness，training backend 換成 `command` 或 `pretrained`。

### 7.1 自訂 training command

```yaml
training:
  backend: command
  command: >
    python train_my_policy.py
    --dataset {dataset_repo_id}
    --output {checkpoint_dir}
    --num-cups {task.num_cups}
    --num-shuffles {task.num_shuffles}
  checkpoint_path: "{checkpoint_dir}/pretrained_model"
```

`run_sweep.py` 會對 `{...}` 做 template replacement。常用 context 包含：

| Template | 內容 |
|----------|------|
| `{variant_name}` | 目前 variant 名稱 |
| `{variant_dir}` | 目前 variant 結果資料夾 |
| `{dataset_file}` | HDF5 dataset 路徑 |
| `{dataset_repo_id}` | LeRobot/HF dataset repo id |
| `{local_dataset_dir}` | 本地 dataset cache/root 路徑 |
| `{checkpoint_dir}` | 此 variant 的 checkpoint 目錄 |
| `{metrics_json}` | 此 variant 的 eval output json |
| `{task.num_cups}` | Flatten 後的 config 欄位也可用 |

### 7.2 使用既有 checkpoint

```yaml
policy:
  backend: lerobot
  type: diffusion
  checkpoint: results/my_existing_run/pretrained_model

training:
  backend: pretrained
```

然後執行：

```bash
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --skip_datagen
```

---

## 8. 輸出目錄

典型 sweep output：

```text
results/exp1_shuffle_scaling/
├── summary.json
├── num_shuffles-0/
│   ├── config.yaml
│   ├── dataset.hdf5
│   ├── checkpoints/
│   │   └── pretrained_model/
│   └── metrics.json
├── num_shuffles-1/
│   └── ...
└── num_shuffles-2/
    └── ...
```

幾種不同資料位置不要混淆：

| 路徑 | 用途 |
|------|------|
| `results/<exp>/<variant>/dataset.hdf5` | sweep variant 的 HDF5 output path |
| `results/<exp>/<variant>/checkpoints/` | training output |
| `results/<exp>/<variant>/metrics.json` | evaluation output |
| `results/<exp>/summary.json` | sweep 彙整指標 |
| `.cache/lerobot/{dataset_repo_id}` | `run_sweep.py` 用來偵測/指定 `--dataset.root` 的專案本地 dataset path |

---

## 9. Debug Checklist

### 9.1 只看環境動畫，不錄製

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --num_demos 3 \
  --num_cups 3 \
  --num_shuffles 2
```

### 9.2 檢查困難設定

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/hard_mode.hdf5 \
  --num_demos 20 \
  --num_cups 5 \
  --num_shuffles 5 \
  --shuffle_speed 2.0
```

### 9.3 只跑 Oracle eval

這是最快確認 phase manager、truth tracking、lift detection 是否合理的方式：

```bash
python scripts/eval_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda \
  --enable_cameras \
  --policy_backend oracle \
  --num_episodes 10 \
  --num_cups 3 \
  --num_shuffles 2 \
  --output_json ./results/debug_oracle_metrics.json
```

Oracle 如果 MSR 很低，優先檢查 Act phase 的 cup dynamics、gripper 接觸、lift threshold。

---

## 10. 常見問題

### Q: 為什麼不用現有 `generate.py`？

`generate.py` 強耦合 UMI pipeline 的 `object_poses.json`，episode 數量由 JSON 驅動。ShellBench 的 episode 由 `--num_demos` 驅動，而且 Reveal/Cover/Shuffle 是 task scripted phase，因此用獨立 `generate_shell_game.py` 比較不會影響既有 pipeline。

### Q: `any_cup_lifted` 會不會讓 DSR 提早結束算錯？

不會。ShellBench 的 selection 是在第一次 cup lift threshold crossed 時記錄 `selected_cup_index`。Episode 之後可以結束，因為 DSR/MSR/SR/kappa 只需要知道「第一次掀的是哪個杯」和「真實藏球杯是哪個」。掀起後不需要再 retreat。

### Q: 可以用通用 `rollout.py` 評估嗎？

不建議。ShellBench 的成功不是 env 內建 `success`，而是專用 evaluator 算 DSR/MSR/SR/kappa。請使用 `scripts/eval_shell_game.py`。

### Q: Phase 1-3 policy 是不是沒有看到？

Evaluation 中 Phase 1-3 會餵 observation 給 LeRobot policy history，但不使用這些 phase 的 action。這樣 Act phase 的 action 能依賴 reveal/shuffle 記憶。

### Q: 為什麼 sweep 的 `extra_args` 是 list？

因為每個 LeRobot CLI override 是一個獨立參數。寫成 list 可以避免 shell quoting 問題：

```yaml
extra_args:
  - "--training.num_epochs=100"
  - "--training.batch_size=32"
```

### Q: 不想 push 到同一個 policy repo 覆蓋怎麼辦？

預設 `policy.repo_id` 會把 variant name 加進 repo id，例如 `alice/shellbench-policy-num_shuffles-2`，所以不同 variant 不會共用同一個 repo。若你手動把 `policy.repo_id` 設成固定值，才可能覆蓋或混在同一個 repo。

### Q: `.cache` 現在涵蓋所有輸出嗎？

不是。`.cache/lerobot/...` 目前只用於本地 dataset reuse 和 training 的 `--dataset.root`。Sweep results、checkpoint、metrics 仍在 `results/...`。HF dataset 若不存在於本地但存在於 Hub，目前由 LeRobot 自己處理下載位置；runner 尚未強制下載到專案 `.cache`。
