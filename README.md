# ShellBench

A parametric shell game benchmark for evaluating memory-aware visuomotor policies, built on NVIDIA Isaac Sim and LeRobot.

Existing visuomotor policies (Diffusion Policy, ACT, etc.) rely on short observation windows and lack long-term memory. The shell game is inherently **non-Markovian** — once the ball is covered, all cups look identical, and the policy must remember where the ball was. ShellBench provides a simulation-based, reproducible, difficulty-controllable test platform for quantifying any manipulation policy's memory capacity.

> **Platform:** Linux only. Requires NVIDIA GPU.

## Task

The task has four phases. Phases 1–3 are scripted by the environment; Phase 4 is controlled by the policy:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Reveal** | Ball appears next to a cup; camera observes for several frames |
| 2 | **Cover** | Cup covers the ball; ball disappears from view |
| 3 | **Shuffle** | Cups swap positions N times |
| 4 | **Act** | Robot lifts a cup (policy-controlled) |

### Difficulty Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_cups` | 3–5 | Number of cups |
| `num_shuffles` | 0–5+ | Number of pairwise swaps |
| `shuffle_speed` | 0.5–2.0 | Speed of cup swaps |

### Metrics

| Metric | Definition |
|--------|------------|
| **DSR** (Decision Success Rate) | Fraction of episodes where the correct cup was selected |
| **MSR** (Manipulation Success Rate) | Fraction of episodes where any cup was successfully lifted |
| **SR** (Success Rate) | Fraction of episodes with correct selection AND successful lift |
| **kappa** (Cohen's Kappa) | Chance-corrected DSR: `(DSR - 1/num_cups) / (1 - 1/num_cups)` |

## Directory Structure

```
Advanced-level/
├── packages/
│   ├── umi/                   # UMI data processing pipeline
│   └── simulator/             # Isaac Lab task definitions (Docker)
├── policies/
│   ├── diffusion/             # Diffusion Policy eval wrapper
│   └── lstm/                  # LSTM Policy (custom LeRobot plugin)
├── scripts/                   # Data generation, evaluation, sweep runner
├── configs/                   # Sweep experiment configs (base + experiments/)
├── outputs/                   # Training & eval results (gitignored)
├── docs/                      # Documentation
│   └── shellbench/            # ShellBench design docs & usage guide
├── tests/                     # Tests
├── data/                      # Data storage (gitignored)
├── Dockerfile / Makefile      # Isaac Sim Docker environment
└── pyproject.toml             # uv workspace root
```

## Setup

### Host (Training)

```bash
uv sync
source .venv/bin/activate
export HF_USER=<your-huggingface-username>
hf auth login
```

### Docker (Simulation / Data Generation / Evaluation)

```bash
make launch-isaaclab       # build & enter Isaac Sim container
```

Inside the container, the repo is mounted at `/workspace/aicapstone`.

## Data Generation

ShellBench uses a dedicated script with FSM-based demonstration generation (not the UMI pipeline).

### Generate LeRobot Dataset

```bash
python scripts/datagen/generate_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --record \
  --use_lerobot_recorder \
  --lerobot_dataset_repo_id "${HF_USER}/shellbench-demo" \
  --dataset_file ./datasets/shell_game.hdf5 \
  --num_demos 100 \
  --num_cups 3 \
  --num_shuffles 2 \
  --seed 42
```

Upload to HF Hub:

```bash
hf upload ${HF_USER}/shellbench-demo \
  .cache/huggingface/lerobot/${HF_USER}/shellbench-demo/
```

### Generate HDF5 Only (No LeRobot Format)

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

## Training

### Diffusion Policy

Diffusion Policy uses LeRobot's built-in `lerobot-train` command directly on the host machine:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/shellbench-demo \
  --policy.type=diffusion \
  --output_dir=outputs/diffusion/shellbench-num_shuffles-2 \
  --job_name=shellbench_diffusion \
  --policy.device=cuda \
  --wandb.enable=true \
  --training.num_epochs=100
```

See [docs/lerobot_training.md](docs/lerobot_training.md) for full flag reference, multi-GPU setup, and troubleshooting.

### LSTM Policy

LSTM Policy is a custom LeRobot plugin with recurrent memory. Training uses a wrapper script that registers the plugin then calls `lerobot-train`:

```bash
python policies/lstm/scripts/train_lstm.py \
  --config policies/lstm/configs/default.yaml
```

Available configs in `policies/lstm/configs/`: `default.yaml`, `num_cups_4.yaml`, `num_cups_5.yaml`, `num_shuffles_0.yaml` through `num_shuffles_5.yaml`, `shuffle_speed_0.5.yaml`, `shuffle_speed_1.5.yaml`, `shuffle_speed_2.0.yaml`.

Resume from a checkpoint:

```bash
python policies/lstm/scripts/train_lstm.py \
  --config outputs/lstm/num_shuffles-3/checkpoints/075000/pretrained_model/train_config.json \
  --resume=true
```

#### Dataset Preparation for LSTM

1. Download the dataset from HF Hub:

   ```bash
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download('johnnyli1220/shellbench-num_shuffles-3',
                     repo_type='dataset',
                     local_dir='.cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-3')
   "
   ```

2. (Recommended) Decode video to images for faster training:

   ```bash
   python policies/lstm/scripts/decode_dataset_to_images.py \
     --src-repo johnnyli1220/shellbench-num_shuffles-3 \
     --src-root .cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-3 \
     --dst-root data/lerobot_img/johnnyli1220/shellbench-num_shuffles-3 \
     --resize 128
   ```

See [policies/lstm/docs/running.md](policies/lstm/docs/running.md) for OOM troubleshooting and tunable hyperparameters.

## Evaluation

### Diffusion Policy

```bash
python policies/diffusion/scripts/eval_diffusion.py \
  --run_dir outputs/diffusion/shellbench-num_shuffles-3 \
  --num_episodes 100 \
  --seed 529
```

The wrapper auto-selects the best checkpoint, reads training config, and infers task parameters from the run directory name.

Evaluate a specific checkpoint step:

```bash
python policies/diffusion/scripts/eval_diffusion.py \
  --run_dir outputs/diffusion/shellbench-num_shuffles-0 \
  --checkpoint_step 60000 \
  --num_episodes 100
```

### LSTM Policy

```bash
python policies/lstm/scripts/eval_lstm.py \
  --run_dir outputs/lstm/num_shuffles-3 \
  --num_episodes 100
```

### Oracle (Upper Bound)

Oracle uses a scripted FSM that knows the ground truth — useful for measuring manipulation difficulty:

```bash
python scripts/eval_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda \
  --enable_cameras \
  --policy_backend oracle \
  --num_episodes 50 \
  --num_cups 3 \
  --num_shuffles 2 \
  --output_json ./results/oracle/metrics.json
```

### Manual Evaluation (Any Policy)

```bash
python scripts/eval_shell_game.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda \
  --enable_cameras \
  --policy_backend lerobot \
  --policy_type lerobot-diffusion \
  --policy_checkpoint_path ./outputs/diffusion/shellbench-num_shuffles-2/checkpoints/pretrained_model \
  --policy_action_horizon 16 \
  --num_episodes 50 \
  --num_cups 3 \
  --num_shuffles 2 \
  --seed 42 \
  --output_json ./results/eval_metrics.json
```

## Sweep Experiments

`scripts/run_sweep.py` reads a base config and a sweep config, expands all parameter combinations, and runs datagen → training → evaluation for each variant.

```bash
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --output_dir results/exp1_shuffle_scaling
```

### Predefined Experiments

```bash
# Exp 1: Fix 3 cups, sweep num_shuffles = 0..5
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml

# Exp 2: Fix 2 shuffles, sweep num_cups = 3,4,5
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp2_num_cups/sweep.yaml

# Exp 3: Fix difficulty, sweep observation_horizon = 2,8,16,32
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp3_obs_horizon/sweep.yaml

# Exp 4: Oracle upper bound (no training)
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp4_oracle/sweep.yaml
```

### Skip Steps

```bash
# Eval only (use existing checkpoints)
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --skip_datagen --skip_training

# Datagen only
python scripts/run_sweep.py \
  --base configs/base.yaml \
  --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml \
  --skip_training --skip_eval
```

## LSTM Smoke Test

Verify the LSTM plugin is correctly installed (registration, BPTT training, inference with memory carry, save/load):

```bash
python policies/lstm/scripts/smoke_test.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [ShellBench Task Description](docs/shellbench/task_description.md) | Task definition and experiment design |
| [ShellBench Usage Guide](docs/shellbench/USAGE.md) | Detailed usage: datagen, training, evaluation, sweep configs |
| [Codebase Overview](docs/shellbench/codebase_overview.md) | Full directory structure and module descriptions |
| [LeRobot Training](docs/lerobot_training.md) | Diffusion Policy training procedure |
| [LSTM Policy — How It Works](policies/lstm/docs/how_it_works.md) | LSTM architecture and integration design |
| [LSTM Policy — Running](policies/lstm/docs/running.md) | LSTM training/eval commands and troubleshooting |
| [Getting Started](docs/getting_started.md) | End-to-end pipeline walkthrough (UMI → datagen → training → eval) |
| [LeRobot Rollout](docs/lerobot_rollout.md) | General policy evaluation in simulator |

## License

MIT — see [LICENSE](LICENSE).
