# ShellBench 實作紀錄

本文件詳細記錄 ShellBench shell game benchmark 的實作過程與每個檔案的功能。

---

## 總覽

根據 `task_description.md` 的任務定義和 `IMPLEMENT.md` 的實作計畫，完成了以下模組：

| 類別 | 檔案 | 狀態 |
|------|------|------|
| Task 環境 | `tasks/shell_game/__init__.py` | 新增 |
| Task 環境 | `tasks/shell_game/shell_game_env_cfg.py` | 新增 |
| Task 環境 | `tasks/shell_game/shell_game_phase_manager.py` | 新增 |
| Task 環境 | `tasks/shell_game/mdp/__init__.py` | 新增 |
| 資料生成 FSM | `datagen/state_machine/shell_game.py` | 新增 |
| 資料生成腳本 | `scripts/datagen/generate_shell_game.py` | 新增 |
| 評估腳本 | `scripts/eval_shell_game.py` | 新增 |
| Sweep 系統 | `scripts/run_sweep.py` | 新增 |
| Config | `configs/base.yaml` | 新增 |
| Config | `configs/experiments/exp1~4/sweep.yaml` | 新增 |
| 註冊 | `tasks/__init__.py` | 修改 |

---

## 詳細說明

### 1. Task 環境 — `shell_game/`

**路徑：** `packages/simulator/src/simulator/tasks/shell_game/`

#### 1a. `__init__.py` — Gym 註冊

以 `HCIS-ShellGame-SingleArm-v0` 名稱向 gymnasium 註冊環境，進入點為 `ShellGameEnvCfg`。模式完全仿照 `CupStacking` 的 `__init__.py`。

#### 1b. `shell_game_env_cfg.py` — 環境設定

**場景設定 (`ShellGameSceneCfg`)：**
- 沿用 Kitchen 場景 (`KITCHEN_CFG`)
- 預定義 5 個杯子 (`cup_0` ~ `cup_4`)，全部使用相同的 `PinkCup.usd`，確保外觀一致
- 未使用的杯子透過 `init_state.pos` 設為 `(0, 0, -10)` 移到場景外
- 1 個球體 (`ball`) 使用 `sim_utils.SphereCfg` 程式化生成（半徑 1.5cm，橘色）
- 杯子和球都設為 `kinematic_enabled=True`，Phase 1-3 期間由 PhaseManager 直接控制位置

**杯子排列邏輯：**
- 杯子沿 x 軸等距排列，以 `x=0.50` 為中心
- 間距 `CUP_SPACING = 0.12m`，根據 `num_cups` 自動計算每個杯子的 x 座標

**Termination 設定 (`TerminationsCfg`)：**
- `time_out`：保留作為安全網，避免 episode 永不結束
- `any_cup_lifted`：自訂 termination function，偵測任一活躍杯子的 z 座標超過 `CUP_Z + 0.05m`
- **沒有 `success` termination**——成功判定（DSR/MSR/SR/κ）交由外部 `ShellGameEvaluator` 計算

**環境參數 (`ShellGameEnvCfg`)：**
- `num_cups`、`num_shuffles`、`shuffle_speed`、`ball_position`、`shell_game_seed`
- 各階段幀數：`reveal_frames=50`、`cover_frames=10`、`shuffle_per_swap_frames=30`、`act_frames=150`
- `object_pose_cfg = None`：不使用 UMI object_poses 系統
- `episode_length_s` 根據階段總步數自動計算

#### 1c. `shell_game_phase_manager.py` — 四階段管理器

這是 ShellBench 最核心的元件，負責驅動 Phase 1-3 的環境動畫，並在 Phase 4 交出控制權。

**四個階段：**

| Phase | 名稱 | 行為 |
|-------|------|------|
| REVEAL | 展示 | 球出現在目標杯子旁邊，camera 可觀察 |
| COVER | 蓋住 | 球被移到場景外（hidden），模擬蓋住效果 |
| SHUFFLE | 洗牌 | 杯子兩兩交換位置 N 次，沿弧形軌跡滑動 |
| ACT | 行動 | 交給 policy/FSM 控制 |

**Shuffle 動畫：**
- 每次 swap 選兩個杯子 `(i, j)`
- 弧形軌跡插值：一個杯子沿 y 正方向偏移，另一個沿 y 負方向，避免碰撞
- 公式：`y_offset = sin(π·t) × arc_height`，`t` 從 0 到 1
- 每次 swap 完成後更新 `cup_init_positions` 和 `ball_cup_idx`（追蹤球的位置）

**Ground Truth 追蹤：**
- `ball_true_cup_index`：球目前在哪個杯子下面（每次 swap 後更新）
- `selected_cup_index`：第一個被提起超過 threshold 的杯子（一旦記錄就不再覆蓋）

**物件位置操控：**
- 使用 `write_root_pose_to_sim()` 逐幀更新杯子和球的位置
- 球的「消失」透過移到 `z=-5.0` 實現（避免 visibility API 不穩定）

#### 1d. `mdp/__init__.py`

匯出 Isaac Lab 和 LeIsaac 的標準 MDP 函式（observation terms、action terms 等），與 template 一致。

---

### 2. State Machine — `shell_game.py`

**路徑：** `packages/simulator/src/simulator/datagen/state_machine/shell_game.py`

**設計邏輯：**
- 繼承 `StateMachineBase`（來自 `leisaac`），與 `CupStackingStateMachine` 相同的介面
- 透過 `set_phase_manager()` 接收 `ShellGamePhaseManager` 的參考，取得 ground truth
- 只負責 Phase 4 (Act) 的動作生成

**Phase 4 的 FSM 階段：**

| Event | 名稱 | 步數 | 動作 |
|-------|------|------|------|
| 0 | MOVE_ABOVE | 160 | 移動 EE 到正確杯子上方（含線性插值起手式） |
| 1 | DESCEND | 80 | 下降到抓取高度 |
| 2 | GRASP | 20 | 閉合 gripper |
| 3 | LIFT | 100 | 提起杯子 |

**IK 控制：**
- 完全複製 `CupStackingStateMachine` 的 Differential IK 邏輯（Damped Least Squares）
- Jacobian 取自 `robot.root_physx_view.get_jacobians()`
- 末端執行器的目標方向固定為 gripper-down

**`get_hold_action()`：**
- 新增方法，Phase 1-3 期間由 `generate_shell_game.py` 呼叫
- 回傳「當前 joint position + gripper open」，維持機器人靜止
- 解決 IMPLEMENT.md 中提到的「hold_action 不能是全零」問題

**`check_success()`：**
- 比對 `phase_manager.selected_cup_index` 和 `ball_true_cup_index`
- 只有選對杯子且成功掀起才算 success

---

### 3. 資料生成腳本 — `generate_shell_game.py`

**路徑：** `scripts/datagen/generate_shell_game.py`

**為何獨立腳本：**
- `generate.py` 強耦合 UMI `object_poses.json`（episode 數量由 JSON 決定）
- ShellBench 的 episode 數量由 `--num_demos` 參數決定
- Phase 1-3 需要特殊的 hold action 處理
- 避免在已穩定的 pipeline 中引入 regression

**流程：**
1. 設定環境參數（`num_cups`、`num_shuffles` 等）
2. 禁用 `time_out` 和 `any_cup_lifted` termination（FSM 控制 episode 結束）
3. 每個 episode：
   - `env.reset()` → `phase_manager.reset(env)` → `sm.reset()`
   - Phase 1-3：`phase_manager.step(env)` 驅動動畫，`env.step(hold_action)` 持續錄 observation
   - Phase 4：`sm.get_action(env)` 驅動掀杯動作
4. `sm.check_success()` 判斷是否保存此 demo

**命令列參數：**
- `--num_demos`：要生成的 demo 數量
- `--num_cups`、`--num_shuffles`、`--shuffle_speed`：shell game 參數
- `--record`、`--dataset_file`：錄製設定
- 支援 `--use_lerobot_recorder` 和 `--resume`

---

### 4. 評估腳本 — `eval_shell_game.py`

**路徑：** `scripts/eval_shell_game.py`

**核心元件：**

#### `ShellGameEvaluator`
- `record_episode()`：記錄每個 episode 的 `selected_cup`、`ball_true`、`dsr`、`msr`、`sr`
- `compute_metrics()`：計算聚合指標
  - **DSR** = 正確選擇率（Decision Success Rate）
  - **MSR** = 成功掀起率（Manipulation Success Rate）
  - **SR** = 完全成功率（選對 + 掀起）
  - **κ** = Cohen's Kappa = `(DSR - chance) / (1 - chance)`，其中 `chance = 1/num_cups`

#### `EvalPolicy`
- 簡化版的 `LeRobotSyncPolicy`（來自 `rollout.py`）
- 載入 LeRobot checkpoint，將 Isaac Lab observation 轉成 LeRobot format 後推論

#### 評估流程
1. 每個 episode：`env.reset()` → `phase_manager.reset()` → `policy.reset()`
2. Phase 1-3：hold action 維持機器人靜止，但 observation 會被餵進 policy history
3. Phase 4：policy 控制，每步呼叫 `phase_manager.update_selection()` 偵測哪個杯子被掀起
4. 偵測到 cup lift 或 timeout 時 episode 結束
5. 輸出 `metrics.json`

---

### 5. Config 管理與 Sweep 系統

#### `configs/base.yaml`
所有預設值的單一來源，涵蓋：
- `task`：杯子數、shuffle 次數、速度
- `phase_duration`：各階段幀數
- `evaluation`：episode 數、threshold
- `policy`：類型、observation horizon
- `data_collection`：方法、demo 數量

#### `configs/experiments/exp*/sweep.yaml`
每個實驗只定義要變動的參數，其餘繼承 `base.yaml`：
- **exp1**：`num_shuffles: [0, 1, 2, 3, 4, 5]`
- **exp2**：`num_cups: [3, 4, 5]`
- **exp3**：`observation_horizon: [2, 8, 16, 32]`
- **exp4**：`policy.type: [oracle]`

#### `scripts/run_sweep.py`
- 讀取 `base.yaml` + `sweep.yaml`
- 用 `itertools.product()` 展開所有參數組合
- 對每個組合依序執行：datagen → training → eval
- 結果存到 `results/<experiment_name>/<variant>/metrics.json`
- 最後輸出 `summary.json` 彙整所有 variant 的指標

---

### 6. Task 註冊

**檔案：** `packages/simulator/src/simulator/tasks/__init__.py`

新增一行 `from . import shell_game`，確保 `import simulator.tasks` 時會觸發 `gym.register()`。

---

## 與現有 Codebase 的整合

### 不修改的檔案
- `scripts/datagen/generate.py`：不動，避免 UMI pipeline regression
- `scripts/rollout.py`：不動，ShellBench 使用專用 `eval_shell_game.py`
- 所有現有 task（cup_stacking、cutlery_arrangement、toy_blocks_collection）

### 設計決策

1. **Termination 與 Success 分離**：`any_cup_lifted` 只代表「有杯子被掀起」（episode 結束），不代表「選對杯子」。真正的成功判定由 `ShellGameEvaluator` 在 episode 結束後計算。

2. **獨立腳本**：`generate_shell_game.py` 和 `eval_shell_game.py` 為獨立入口，不修改現有 `generate.py` / `rollout.py`。

3. **Kinematic 物件**：杯子和球在 Phase 1-3 期間為 kinematic（不受物理引擎影響），由 `write_root_pose_to_sim()` 直接控制位置。

4. **Hold Action**：Phase 1-3 的 hold action 使用「當前 joint position + gripper open」，而非全零 action。

5. **單環境限制**：第一版只支持 `num_envs=1`。若未來要支持多環境，PhaseManager 的所有狀態需改為 per-env tensor。

---

## 檔案結構

```
packages/simulator/src/simulator/
├── tasks/
│   ├── __init__.py                              # 已修改：新增 shell_game import
│   └── shell_game/
│       ├── __init__.py                          # gym.register
│       ├── shell_game_env_cfg.py                # 環境設定 + termination
│       ├── shell_game_phase_manager.py          # 四階段 FSM
│       └── mdp/
│           └── __init__.py                      # MDP re-exports
├── datagen/state_machine/
│   └── shell_game.py                            # 資料生成用 FSM planner

scripts/
├── datagen/
│   └── generate_shell_game.py                   # 資料生成入口
├── eval_shell_game.py                           # 評估入口
└── run_sweep.py                                 # 實驗 sweep runner

configs/
├── base.yaml                                    # 預設參數
└── experiments/
    ├── exp1_shuffle_scaling/sweep.yaml
    ├── exp2_num_cups/sweep.yaml
    ├── exp3_obs_horizon/sweep.yaml
    └── exp4_oracle/sweep.yaml
```
