# ShellBench Implementation Plan

基於現有 codebase 架構，將 shell game benchmark 實作為一個新的 HCIS task。以下說明每個模組要改什麼、新增什麼、為什麼這樣設計。

---

## 現有架構分析

現有 codebase 的 task 建立模式非常一致：

```
tasks/<task_name>/
├── __init__.py                  # gym.register()
├── <task_name>_env_cfg.py       # 繼承 SingleArmFrankaTaskEnvCfg，定義 scene + termination
└── mdp/terminations.py          # （可選）自訂 termination function

datagen/state_machine/
└── <task_name>.py               # 繼承 StateMachineBase，定義 scripted demo 的 FSM

scripts/datagen/generate.py      # TASK_REGISTRY 映射 task → StateMachine，驅動資料生成
scripts/rollout.py               # 載入 policy checkpoint 做 evaluation
```

**ShellBench 與現有 task 的關鍵差異：**

| 面向 | 現有 task (CupStacking 等) | ShellBench |
|------|---------------------------|------------|
| 物件數量 | 固定 (2-3 個) | 可變 (3/4/5 個杯子 + 1 球) |
| 任務流程 | 全程 policy/FSM 控制 | Phase 1-3 環境 scripted，Phase 4 才是 policy |
| 物件位置來源 | UMI object_poses.json | 參數化隨機生成 |
| 動畫需求 | 無 | 杯子 shuffle 滑動動畫 |
| 評估指標 | 二元 success/fail | DSR / MSR / SR / κ 四種指標 |
| 實驗控制 | 無 sweep 機制 | base.yaml + sweep.yaml 系統 |

---

## 實作步驟

### Step 1: 建立 Assets

**目標：** 準備 shell game 需要的 3D 物件。

**路徑：** `packages/simulator/assets/scenes/kitchen/objects/`

- **杯子：** 直接複用現有 `PinkCup/PinkCup.usd`（或 `BlueCup/BlueCup.usd`），統一外觀。ShellBench 需要所有杯子長一樣，所以全部使用同一個 USD，在 scene config 中以不同 prim path 生成多個 instance。
- **球：** 用 Isaac Sim 的 `sim_utils.SphereCfg` 程式化生成即可（半徑 ~1.5cm），不需額外 USD 檔。或者也可以用現有 `Orange001.usd` 作為替代。
- **場景：** 沿用 kitchen scene（`KITCHEN_CFG`）。

**需要先驗證的 asset 細節：**
- Shell game 需要「倒扣杯子」，不能只假設現有杯子 USD 直接可用。需確認 init rotation、visual origin、collision shape 是否能穩定遮球、被夾起、且不會和球或桌面穿模。
- 如果現有杯子 collision 不適合倒扣抓取，第一版可以仍複用 visual USD，但要補 collision approximation 或調整 spawn pose。
- 同一個杯子 USD 可以 spawn 多次到不同 prim path；球體用 primitive shape 即可。

---

### Step 2: 新增 Task — `shell_game`

**路徑：** `packages/simulator/src/simulator/tasks/shell_game/`

```
shell_game/
├── __init__.py                    # gym.register("HCIS-ShellGame-SingleArm-v0", ...)
├── shell_game_env_cfg.py          # ShellGameEnvCfg — 核心環境設定
└── shell_game_phase_manager.py    # ShellGamePhaseManager — 管理 Reveal/Cover/Shuffle/Act 四階段
```

#### 2a. `__init__.py` — Gym 註冊

```python
import gymnasium as gym

gym.register(
    id="HCIS-ShellGame-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shell_game_env_cfg:ShellGameEnvCfg",
    },
)
```

#### 2b. `shell_game_env_cfg.py` — 環境設定

仿照 `CupStackingEnvCfg` 的模式，但有以下關鍵不同：

**Scene：**
- 根據 `num_cups` 參數動態生成 N 個同外觀的杯子（`cup_0`, `cup_1`, ..., `cup_{n-1}`）
- 等距排列於桌面（x 軸方向排列，間距根據 num_cups 自動計算）
- 1 個球體物件（`ball`），初始放在隨機杯子旁
- 由於 Isaac Lab 的 `@configclass` 不支持動態欄位，需要預定義最大數量的杯子（max 5），並用 `init_state` 把不需要的杯子移到場景外

```python
@configclass
class ShellGameSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # 預定義 5 個杯子（最大支持數量），不用的會被移到場景外
    cup_0: RigidObjectCfg = RigidObjectCfg(...)
    cup_1: RigidObjectCfg = RigidObjectCfg(...)
    cup_2: RigidObjectCfg = RigidObjectCfg(...)
    cup_3: RigidObjectCfg = RigidObjectCfg(...)  # num_cups < 4 時移出場景
    cup_4: RigidObjectCfg = RigidObjectCfg(...)  # num_cups < 5 時移出場景

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/ball",
        spawn=sim_utils.SphereCfg(radius=0.015, ...),  # 或用 Orange USD
    )
```

**Termination：**
- 不使用現有的 `success` termination（因為「選對杯子」的成功判定需要在 evaluation 中做，不在 env 內做）
- 保留 `time_out` 作為 Phase 4 的步數上限（保底，避免 policy 卡住時 episode 永遠不結束）
- 新增 `any_cup_lifted` termination：任一杯子的 z 座標超過初始高度 + 門檻（如 3-5cm）時，提前結束 episode。這代表「policy 已做出選擇 / 動作已完成」，不是「任務成功」。
- `any_cup_lifted` 必須避免被 generic `rollout.py` 誤解成 success。ShellGame 的正確評估只能由 `eval_shell_game.py` 計算：掀起杯子只代表 MSR/selection，掀起正確杯子才代表 DSR/SR。
- evaluator 必須在 termination 發生的同一個 step 記錄 `selected_cup`（哪個杯子第一次超過 lift threshold，或哪個杯子最先被 gripper contact）。不要等 episode reset 或 retreat 後才用 end-effector 最終位置判斷。

```python
@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    any_cup_lifted = DoneTerm(
        func=any_cup_z_above_threshold,
        params={
            "cup_cfgs": [SceneEntityCfg(f"cup_{i}") for i in range(5)],
            "lift_threshold": 0.03,
        },
    )
    # 注意：沒有 success termination，成功判定交給外部 evaluator
```

**新增 Config 欄位：**

```python
@configclass
class ShellGameEnvCfg(SingleArmFrankaTaskEnvCfg):
    # Shell game 參數
    num_cups: int = 3
    num_shuffles: int = 2
    shuffle_speed: float = 1.0
    ball_position: str = "random"  # "random" 或指定 cup index
    shell_game_seed: int = 42
    # 第一版明確限制單環境。若未來要支援 num_envs > 1，
    # PhaseManager 的 truth、shuffle sequence、selected_cup 都必須 tensor 化。
    supported_num_envs: int = 1

    # 各階段持續幀數
    reveal_frames: int = 50
    cover_frames: int = 10
    shuffle_per_swap_frames: int = 30
    act_frames: int = 150
```

**不使用 UMI object_poses：** ShellBench 不需要從真實世界讀取物件姿態，`object_pose_cfg` 設為 `None`。杯子排列位置由 `num_cups` 參數化計算。

#### 2c. `shell_game_phase_manager.py` — 四階段 FSM

這是 ShellBench 最獨特的元件。現有 task 不需要「環境主動動物件」的機制，ShellBench 需要。

```python
class ShellGamePhaseManager:
    """管理 Shell Game 的四個階段。
    
    Phase 1 (Reveal):  球出現在某個杯子旁邊，camera 可觀察
    Phase 2 (Cover):   杯子蓋住球，球消失
    Phase 3 (Shuffle): 杯子兩兩交換 N 次
    Phase 4 (Act):     交給 policy 控制
    """

    def __init__(self, cfg: ShellGameEnvCfg): ...

    def reset(self, env, rng): ...
        # 決定球放在哪個杯子下面
        # 生成 shuffle sequence（哪兩個杯子交換）
        # 重置所有內部狀態

    def step(self, env) -> bool:
        # 根據當前 phase 執行對應邏輯
        # 返回 is_act_phase（True 時交給 policy）

    @property
    def ball_true_cup_index(self) -> int:
        # Ground truth: 球在哪個杯子下面；第一版只支援 num_envs=1

    @property
    def cup_positions(self) -> list[tuple[float, float, float]]:
        # 單一 env 中每個杯子當前的世界座標

    @property
    def selected_cup_index(self) -> int | None:
        # 第一次觸發 contact/lift threshold 的杯子；一旦記錄就不再覆蓋
```

**Shuffle 動畫實作方式：**
- 每次 shuffle 選兩個杯子 (i, j)
- 沿弧形軌跡 (圓弧插值) 交換位置，避免兩杯碰撞
- 優先使用 codebase 既有的 `RigidObject.write_root_pose_to_sim(...)` 逐幀更新杯子位置；`set_world_pose()` 若要使用，必須先在 Isaac Lab 中 spike 驗證 API 存在且可逐步更新 rigid object
- 球在 Cover 階段後設為 hidden。若 `set_visibility(False)` per-step 切換不可行，改用把球移到桌面下方且關閉碰撞的方式
- 內部維護 `cup_to_ball` mapping，每次 swap 後更新
- 每個 step 檢查 cup lift/contact；第一次偵測到選擇時寫入 `selected_cup_index`，供 evaluator 計算 DSR/MSR/SR

**弧形軌跡計算：**
```
P_mid = (P_i + P_j) / 2
offset = perpendicular_direction * arc_height
t = step / total_steps  (0→1)
P_i(t) = lerp(P_i, P_j, t) + sin(π*t) * offset   # 一個往上繞
P_j(t) = lerp(P_j, P_i, t) - sin(π*t) * offset   # 另一個往下繞
```

**PhaseManager 與 env 的整合方式：**
- 在 `generate_shell_game.py` 和 `eval_shell_game.py` 的 step loop 中，先呼叫 `phase_manager.step(env)`
- 若返回 `is_act_phase=False`（Phase 1-3），仍然要呼叫 `env.step(hold_action)`，讓 camera observation 被 recorder / policy history 看見
- `hold_action` 不能是全零 action。Franka keyboard action 是 joint position target，所以 hold action 必須使用「目前 7 軸 joint position + 目前/預設 gripper command」，避免機器人在 reveal/shuffle 時被命令回 0
- 若返回 `is_act_phase=True`（Phase 4），才讓 policy/FSM 生成 action
- evaluation 時 Phase 1-3 雖然不套用 policy action，但仍要把每一步 observation 餵進 policy/preprocessor/history queue；否則 policy 到 Act phase 只看到蓋住後的畫面，benchmark 就測不到 memory

---

### Step 3: 新增 State Machine — `ShellGameStateMachine`

**路徑：** `packages/simulator/src/simulator/datagen/state_machine/shell_game.py`

用於生成 demo 資料。此 FSM planner **知道 ground truth**（球在哪），所以永遠選對杯子。

```
Phase 4 的 FSM 階段:
State 1: MOVE_ABOVE     → 移到正確杯子正上方
State 2: DESCEND_GRASP  → 下降並夾住杯子
State 3: LIFT           → 提起杯子（露出球）
```

- 完全仿照 `CupStackingStateMachine` 的 IK + waypoint 模式
- 目標杯子由 `PhaseManager.ball_true_cup_index` 決定。實作時需明確資料流：建議由 `generate_shell_game.py` 建立 `phase_manager`，再傳入 `ShellGameStateMachine(phase_manager)`；或把 phase manager 掛在 env wrapper 上。不要讓 state machine 隱式查全域狀態。
- FSM 不做 RETREAT。Shell game 的 demo 目標是「選對並掀起杯子」，杯子一達到 lift threshold 即 episode done，避免後續 retreat 汙染 DSR 判斷或讓資料包含與 benchmark 無關的動作。
- `check_success()`：判斷 `phase_manager.selected_cup_index == phase_manager.ball_true_cup_index`

---

### Step 4: 修改資料生成流程

**檔案：** `scripts/datagen/generate_shell_game.py`

由於 ShellBench 不使用 UMI `object_poses.json`，第一版採用獨立生成腳本，不修改現有 `scripts/datagen/generate.py` 的 UMI replay flow。

```python
# 不需要 --object_poses 參數
# 新增參數：
#   --num_cups, --num_shuffles, --shuffle_speed
#   --num_demos (取代 object_poses 驅動的 episode 數)
#   --seed

# 每個 episode 的流程：
# 1. env.reset()
# 2. PhaseManager.reset() → 隨機決定球位置、shuffle 序列
# 3. Phase 1-3: PhaseManager.step(env) 驅動環境動畫，env.step(hold_action) 持續錄 observation
# 4. Phase 4: StateMachine.get_action(env) 驅動機器人掀杯
# 5. 杯子首次超過 lift threshold 時記錄 selected_cup，episode done
# 6. check_success() → 決定是否保存此 demo
```

**選擇獨立腳本而非修改 generate.py 的理由：**
- `generate.py` 強耦合 UMI object_poses flow（episode 數量由 JSON 驅動）
- ShellBench 的 episode 數量由 `--num_demos` 驅動，Phase 1-3 有環境動畫
- ShellBench 需要特殊的 hold action、policy history/recorder handling、selected_cup 記錄；放在獨立腳本比較不會破壞既有 pipeline
- 獨立腳本避免在已穩定的 pipeline 中引入 regression

---

### Step 5: 修改 Evaluation 流程

**目標：** 在 rollout 基礎上加入 ShellBench 專用指標。

**方案：新增 `scripts/eval_shell_game.py`**

這個腳本與 `rollout.py` 結構類似，但加入：

#### 5a. PhaseManager 整合

```python
# 與 datagen 相同，Phase 1-3 由 PhaseManager 驅動
# Phase 4 交給 policy
# 但 policy 不知道 ground truth
```

#### 5b. Evaluation Metrics

```python
class ShellGameEvaluator:
    def __init__(self, num_cups: int):
        self.results = []

    def record_episode(self, env, phase_manager):
        # selected_cup 必須是在 contact/lift threshold 首次觸發當下記錄的杯子
        selected_cup = phase_manager.selected_cup_index
        true_cup = phase_manager.ball_true_cup_index
        msr = selected_cup is not None
        dsr = msr and (selected_cup == true_cup)
        sr = dsr and msr

        self.results.append({...})

    def compute_metrics(self) -> dict:
        dsr = mean([r["dsr"] for r in self.results])
        msr = mean([r["msr"] for r in self.results])
        sr = mean([r["sr"] for r in self.results])
        chance = 1.0 / self.num_cups
        # 保留 below-chance 的負值，不截斷為 0，才符合 Cohen's kappa / chance-corrected score
        kappa = (dsr - chance) / (1.0 - chance)
        return {"DSR": dsr, "MSR": msr, "SR": sr, "kappa": kappa}

    def update_selection(self, env, phase_manager):
        # 每個 step 呼叫，優先用 contact detection 記錄第一個被 gripper 接觸的杯子
        # fallback: 記錄第一個 root z > init_z + lift_threshold 的杯子
        # 若多個杯子同時觸發，選「超過 threshold 最多」或「最早觸發」者
        # 一旦 phase_manager.selected_cup_index 已有值，就不再覆蓋
```

#### 5c. 結果輸出

每次 evaluation 輸出 JSON：

```json
{
  "config": {"num_cups": 3, "num_shuffles": 2, "shuffle_speed": 1.0},
  "num_episodes": 50,
  "DSR": 0.42,
  "MSR": 0.88,
  "SR": 0.37,
  "kappa": 0.13,
  "per_episode": [...]
}
```

---

### Step 6: Config 管理與 Sweep 系統

**路徑：** `configs/`

```
configs/
├── base.yaml
└── experiments/
    ├── exp1_shuffle_scaling/sweep.yaml
    ├── exp2_num_cups/sweep.yaml
    ├── exp3_obs_horizon/sweep.yaml
    └── exp4_oracle/sweep.yaml
```

**新增 `scripts/run_sweep.py`：**

```python
# 讀取 base.yaml + sweep.yaml
# 展開所有參數組合
# 對每個組合：
#   1. 生成 demo 資料（呼叫 generate_shell_game.py）
#   2. 訓練 policy（呼叫 lerobot-train）
#   3. 評估（呼叫 eval_shell_game.py）
#   4. 存結果到 results/<experiment_name>/<variant>/metrics.json
```

---

### Step 7: Tasks 註冊

**檔案：** `packages/simulator/src/simulator/tasks/__init__.py`

```python
from . import cup_stacking
from . import cutlery_arrangement
from . import toy_blocks_collection
from . import shell_game           # 新增
```

---

## 檔案變更清單

### 新增檔案

| 檔案 | 用途 |
|------|------|
| `packages/simulator/src/simulator/tasks/shell_game/__init__.py` | Gym 註冊 |
| `packages/simulator/src/simulator/tasks/shell_game/shell_game_env_cfg.py` | 環境設定 |
| `packages/simulator/src/simulator/tasks/shell_game/shell_game_phase_manager.py` | 四階段 FSM |
| `packages/simulator/src/simulator/datagen/state_machine/shell_game.py` | Demo 生成用 FSM planner |
| `scripts/datagen/generate_shell_game.py` | 資料生成入口 |
| `scripts/eval_shell_game.py` | 評估入口 |
| `scripts/run_sweep.py` | 實驗 sweep runner |
| `configs/base.yaml` | 基礎參數設定 |
| `configs/experiments/exp*/sweep.yaml` | 各實驗的 sweep 設定 |

### 修改檔案

| 檔案 | 變更 |
|------|------|
| `packages/simulator/src/simulator/tasks/__init__.py` | 加入 `from . import shell_game` |

> 第一版不修改 `scripts/datagen/generate.py` 或 `scripts/rollout.py`。ShellBench 使用專用 `generate_shell_game.py` / `eval_shell_game.py`，避免把 selection/done 語意混進既有 success/fail pipeline。

---

## 實作順序建議

按照依賴關係，建議以下順序：

```
Phase A — 環境建立（可先跑通最基礎的場景）
  1. shell_game_env_cfg.py（Scene + 杯子 + 球）
  2. shell_game/__init__.py（gym 註冊）
  3. tasks/__init__.py（加入 import）
  → 驗證：能跑 gym.make("HCIS-ShellGame-SingleArm-v0")，看到杯子和球

Phase B — 環境動畫
  4. shell_game_phase_manager.py（Reveal / Cover / Shuffle 動畫邏輯）
  → 驗證：env reset 後能看到球出現、蓋住、杯子交換動畫

Phase C — 資料生成
  5. shell_game.py（StateMachine — Phase 4 掀杯 FSM）
  6. generate_shell_game.py（資料生成腳本）
  → 驗證：能生成 LeRobot format 的 demo dataset

Phase D — 訓練 & 評估
  7. 用 lerobot-train 訓練 policy
  8. eval_shell_game.py（評估腳本 + metrics）
  → 驗證：能跑完 evaluation 並輸出 metrics JSON

Phase E — 實驗系統
  9. configs/ + run_sweep.py
  → 驗證：能一鍵跑完 exp1 所有參數組合
```

---

## 技術風險與備案

### 風險 1: `@configclass` 不支持動態數量的物件

Isaac Lab 的 `@configclass` 要求 scene 欄位在 class 定義時就確定。無法根據 `num_cups` 動態新增欄位。

**備案：** 預定義 5 個杯子（`cup_0` ~ `cup_4`），不使用的杯子移到場景外（z=-10）或設為 invisible。這是 Isaac Lab 專案中的標準做法。

### 風險 2: Shuffle 動畫中的物理碰撞

杯子在 shuffle 時逐幀強制移動，可能與物理引擎衝突；另外 `set_world_pose()` 不是目前 codebase 已使用的 API。

**備案：** 優先使用現有 pattern `RigidObject.write_root_pose_to_sim(...)`。Shuffle 期間暫時關閉杯子的物理模擬（`disable_gravity=True`, kinematic mode），shuffle 完畢後恢復。或者讓杯子在整個 Phase 1-3 都是 kinematic，Phase 4 才啟用 rigid body physics。

### 風險 3: 球的可見性切換

Isaac Sim 的 visibility 切換在 `set_visibility()` API 是否支援 per-step 切換？

**備案：** 若 visibility 切換有問題，改用將球移到桌面下方（z=-1）來實現「消失」效果。

### 風險 4: Phase 1-3 期間 camera observation 的 recording

Phase 1-3 期間機器人不動，但 camera 需要持續錄影（觀察 shuffle 過程），這些 observation 對 policy 的 memory 能力至關重要。

**方案：** Phase 1-3 期間照常 `env.step(hold_action)`，這樣 recorder 會持續錄製 observation。`hold_action` 必須是目前 joint position target，不是全零 action。Evaluation 時也要把 Phase 1-3 observation 餵進 policy history/preprocessor queue，但不套用 policy 輸出的 action。

### 風險 5: `any_cup_lifted` 被誤當成 success

現有 generic `rollout.py` 會把 termination 視為 success，但 ShellGame 的 `any_cup_lifted` 只代表 selection/done，不代表選對。

**備案：** 第一版只用 `eval_shell_game.py` 跑 ShellBench 指標。`any_cup_lifted` / `selection_made` 不命名為 `success`，且 evaluator 在同一個 step 記錄 `selected_cup` 後再計算 DSR/MSR/SR。

### 風險 6: 多環境 parallel rollout 尚未設計

目前 PhaseManager 介面以單一 `ball_true_cup_index`、單一 `selected_cup_index`、單一 cup positions list 為主，無法正確表示 `num_envs > 1` 時每個 env 各自的 truth / shuffle sequence / selected cup。

**備案：** 第一版 ShellBench 明確限制 `num_envs=1`。若未來要支援 parallel env，PhaseManager 需要把 truth、shuffle sequence、cup positions、selected cup 全部改為 per-env tensor/list，metrics 也要逐 env 聚合。

---

## 與 Chameleon 的互補定位

| 面向 | Chameleon (Guo et al., 2026) | ShellBench (Ours) |
|------|------------------------------|-------------------|
| 平台 | Real robot | Isaac Sim simulation |
| 難度 | 固定 | 參數化可調 |
| 可復現性 | 受限於物理環境 | 完全可復現 (seed) |
| 規模 | 受限於硬體 | 可大規模平行化 |
| Metric | 類似 | DSR / MSR / SR / κ（相容） |
