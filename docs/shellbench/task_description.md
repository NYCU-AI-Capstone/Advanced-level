# ShellBench: A Parametric Shell Game Benchmark for Memory-Aware Manipulation

## Motivation

現有的 visuomotor policy（如 Diffusion Policy、ACT）大多只看最近幾幀 observation 來決定動作，缺乏 long-term memory。Shell game（猜杯子遊戲）天然是一個 **non-Markovian** 任務——球被蓋住後，三個杯子外觀相同，光看當前畫面無法判斷球在哪，policy 必須記得之前發生的事。

我們在 Isaac Sim 中建立一個**難度可參數化調整**的 shell game 環境，作為量化任意 manipulation policy 記憶能力的 benchmark。與 Chameleon (Guo et al., 2026) 的 real-robot + 固定難度設定互補，我們提供 simulation-based、可復現、可控難度的測試平台。

---

## Task Definition

### 任務流程

任務分為四個階段，其中 Phase 1–3 為 scripted（環境自動執行），Phase 4 為 policy 控制：

| Phase | 名稱 | 內容 | 控制方式 |
|-------|------|------|---------|
| 1 | **Reveal** | 球出現在某個杯子旁邊，持續數幀讓 camera 觀察 | 環境 scripted |
| 2 | **Cover** | 杯子蓋住球，球從畫面中消失 | 環境 scripted |
| 3 | **Shuffle** | 杯子兩兩交換位置 N 次 | 環境 scripted |
| 4 | **Act** | 機器人掀開一個杯子 | **Policy 控制** |

### 可控難度參數

```yaml
shell_game:
  num_cups: 3            # 杯子數量：3 / 4 / 5
  num_shuffles: 2        # 交換次數：1 / 2 / 3 / 4 / 5+
  shuffle_speed: 1.0     # 交換速度：0.5（慢）/ 1.0（中）/ 2.0（快）
  ball_position: random  # 球的初始位置：random 或指定
  seed: 42               # 隨機種子，確保可復現
```

---

## Implementation Plan

### 1. 參數化 Isaac Sim 環境

**場景設定：**
- 在 Entry level 的 kitchen scene 基礎上修改
- 放置 N 個**相同外觀**的倒扣杯子（使用 Entry 的杯子 asset，統一顏色）
- 放置 1 個小球（從 USD Assets 取得或自建簡單球體）
- 杯子等距排列於桌面，位置根據 `num_cups` 自動計算

**Shuffle 邏輯：**
- 每次 shuffle 隨機選兩個杯子交換位置
- 杯子沿弧形軌跡滑動（避免碰撞），使用 `set_world_pose()` 逐幀設定位置
- 球在 Cover 階段設為 invisible，不需要模擬球的物理碰撞
- 環境內部維護 ground truth（球在哪個杯子下面）

**Phase 管理：**
- 用 FSM 管理四個階段的轉換
- 每個階段的持續時間可配置
- Phase 1–3 期間機器人靜止不動，camera 持續錄影

### 2. 改寫 FSM Planner（資料生成）

基於 Entry level 的 pick-and-place FSM planner 改寫：

```
State 1: IDLE           → Phase 1-3 期間，機器人靜止，等待 shuffle 完成
State 2: MOVE_ABOVE     → 移到正確杯子正上方（planner 知道 ground truth）
State 3: DESCEND_GRASP  → 下降並夾住杯子
State 4: LIFT           → 提起杯子（露出球）
State 5: RETREAT        → 手臂退回初始位置
```

FSM Planner 因為知道球的位置（作弊），所以永遠生成正確的 demo。這些 demo 錄製成 LeRobot format dataset 後用於訓練 policy。

### 3. Evaluation Function

**指標定義（參考 Chameleon 的 metric 設計）：**

| 指標 | 定義 | 意義 |
|------|------|------|
| **DSR** (Decision Success Rate) | 機器人是否選對杯子 | 測記憶能力 |
| **MSR** (Manipulation Success Rate) | 機器人是否成功掀起杯子（不論對錯） | 測操作能力 |
| **SR** (Success Rate) | 機器人是否選對並成功掀起杯子 | 整體成功率 |
| **κ** (Cohen's Kappa) | (DSR − chance) / (1 − chance) | 校正隨機猜對的因素 |

**判定方式：**

DSR 判定（機器人選了哪個杯子）：
- 優先使用 Isaac Sim 的 contact detection API，判斷 gripper 實際接觸了哪個杯子
- 若無接觸資訊，則以 end-effector 最終位置與各杯子的距離判定，取最近者
- 設定最低距離門檻（如 5cm）：若 end-effector 與所有杯子的距離都超過門檻，判定為「未選擇」，視為失敗
- 這避免了機器人完全不動或停在中間時被誤判為「選了最近的杯子」

MSR 判定（是否成功掀起杯子）：
- 判斷杯子的 z 座標是否比初始高度高出一定門檻（如 3cm）
- 不論掀的是哪個杯子，只要有成功掀起即算 MSR 通過

Chance level：
- 1/num_cups（3 杯 = 33.3%，4 杯 = 25%，5 杯 = 20%）

---

## Config 管理與實驗控制

### 結構：一個 base config + 實驗用 override

```
configs/
├── base.yaml                    # 所有預設值，只維護這一份
├── experiments/
│   ├── exp1_shuffle_scaling/    # Exp 1: 改 shuffle 次數
│   │   ├── sweep.yaml           # 定義這個實驗要掃哪些變數
│   │   └── results/             # 跑完的結果自動存這裡
│   ├── exp2_num_cups/           # Exp 2: 改杯子數量
│   │   ├── sweep.yaml
│   │   └── results/
│   ├── exp3_obs_horizon/        # Exp 3: 改 observation horizon
│   │   ├── sweep.yaml
│   │   └── results/
│   └── exp4_oracle/             # Exp 4: oracle upper bound
│       ├── sweep.yaml
│       └── results/
```

### base.yaml — 所有預設值

```yaml
# configs/base.yaml
task:
  num_cups: 3
  num_shuffles: 2
  shuffle_speed: 1.0
  ball_position: random
  
phase_duration:              # 各階段持續幀數
  reveal: 50
  cover: 10
  shuffle_per_swap: 30       # 每次交換的動畫幀數
  act: 150

evaluation:
  num_episodes: 50           # 每個設定跑幾次
  dsr_distance_threshold: 0.05
  msr_lift_threshold: 0.03
  seed: 42

policy:
  type: diffusion
  observation_horizon: 2     # Diffusion Policy 看幾幀
  checkpoint: null

data_collection:
  method: fsm_planner        # fsm_planner / keyboard
  num_demos: 100
```

### sweep.yaml — 每個實驗只寫「要改什麼」

```yaml
# configs/experiments/exp1_shuffle_scaling/sweep.yaml
experiment_name: exp1_shuffle_scaling
description: "固定 3 杯中速，改變 shuffle 次數，觀察 DSR 衰減"

sweep:
  task.num_shuffles: [0, 1, 2, 3, 4, 5]

# 其他所有參數繼承 base.yaml
```

```yaml
# configs/experiments/exp2_num_cups/sweep.yaml
experiment_name: exp2_num_cups
description: "固定 2 次 shuffle 中速，改變杯子數量"

sweep:
  task.num_cups: [3, 4, 5]
```

```yaml
# configs/experiments/exp3_obs_horizon/sweep.yaml
experiment_name: exp3_obs_horizon
description: "固定難度，改變 policy 的 observation horizon"

sweep:
  policy.observation_horizon: [2, 8, 16, 32]
```

### 跑實驗的方式

寫一個 `run_sweep.py`，讀 base.yaml + sweep.yaml，自動展開所有組合：

```bash
# 跑單一設定（debug 用）
python run.py --config configs/base.yaml --override task.num_shuffles=3

# 跑整個實驗（自動掃所有變數）
python run_sweep.py --base configs/base.yaml \
                    --sweep configs/experiments/exp1_shuffle_scaling/sweep.yaml

# 內部會自動展開成：
#   run 1: base + num_shuffles=0 → 結果存到 results/exp1_shuffle_scaling/ns0/
#   run 2: base + num_shuffles=1 → 結果存到 results/exp1_shuffle_scaling/ns1/
#   run 3: base + num_shuffles=2 → ...
#   ...
```

### 結果輸出格式

每次 run 自動產生一個 JSON：

```json
// results/exp1_shuffle_scaling/ns2/metrics.json
{
  "config": {"num_cups": 3, "num_shuffles": 2, "shuffle_speed": 1.0},
  "num_episodes": 50,
  "DSR": 0.42,
  "MSR": 0.88,
  "SR": 0.37,
  "kappa": 0.13,
  "per_episode": [
    {"episode": 0, "selected_cup": 1, "ball_true": 2, "dsr": false, "msr": true},
    ...
  ]
}
```

---

## Experiment Design

用此 benchmark 跑多個 policy，畫出 **memory capacity curve**：

**Exp 1 — Shuffle 次數 vs. 成功率**（核心實驗）
> 固定 3 杯、中速，shuffle 次數從 0 到 5，比較各方法的 DSR 衰減曲線

**Exp 2 — 杯子數量 vs. 成功率**
> 固定 2 次 shuffle、中速，杯子數從 3 到 5

**Exp 3 — Observation Horizon 的影響**
> 固定難度，改變 Diffusion Policy 的 observation horizon（2 / 8 / 16 / 32 frames）

**Exp 4 — Oracle Upper Bound**
> 將 ground truth（球在哪）直接作為 observation 輸入，測任務本身的 manipulation 難度上限

---

## 預期成果

1. 一個可配置、可復現的 simulation benchmark（含 code + config + evaluation script）
2. 不同難度下多種 policy 的 memory capacity curve
3. 對「policy 需要什麼程度的記憶能力才能解 shell game」的量化分析