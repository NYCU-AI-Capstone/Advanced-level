# ShellBench LSTM Policy — Implementation Plan

> 目標：在 ShellBench 上實作一個 **recurrent (LSTM) imitation-learning policy**，
> 作為「真的有跨整段 episode memory」的對照組，跟 built-in 的 ACT / Diffusion / VQ-BeT
> 比較記憶能力曲線（DSR vs. num_shuffles / num_cups）。
>
> 定位來源：`documents/training_policy_survey.md` §6.1 #2（Robomimic BC-RNN / LSTM-GMM 風格）。

本文件是「動手寫 code 前的藍圖」。實際程式碼會放在 `Advanced-level/LSTM/` 底下。

---

## 0. 為什麼是 LSTM（一句話）

ShellBench 是 non-Markovian：決定答案的資訊（球在哪個杯子）只在 Phase 1 (Reveal) 出現，
之後蓋住、洗牌，policy 要到 Phase 4 (Act) 才動作，中間隔了 **200~360 幀**。
LSTM 用一個 hidden state（「腦中的筆記本」）把 Reveal 看到的資訊一路帶到 Act，
這正是 built-in policy（只看有限視窗）做不到的事。

---

## 1. 關鍵事實：LSTM 不是 LeRobot built-in

ACT / Diffusion / VQ-BeT 只要改 `--policy.type` flag 就能跑；**LSTM 沒有**，
必須自己寫一個 LeRobot custom policy plugin，讓兩個地方都認得它：

- **訓練端**：`lerobot-train --policy.type=<我們的名字>`
- **評估端**：`scripts/eval_shell_game.py` 裡的 `get_policy_class(<我們的名字>)`

一個 LeRobot policy 由兩個 class 組成：
- `PreTrainedConfig` 子類（超參數，用 `--policy.xxx` 設定）
- `PreTrainedPolicy` 子類（模型本體：`forward()` 訓練、`select_action()` 推論）

---

## 2. 模型架構（第一版，刻意簡單）

```
                       ┌─ front camera (主要記憶來源) ─┐
  每一幀 observation ──┤  wrist camera                  ├─→ image encoder (ResNet18)
                       └─ joint state (7 dim)            ┘        │
                                                                  ▼
                                          [影像特徵 + 關節特徵] 串接成一個向量
                                                                  │
                                                                  ▼
                                                    ┌──────────────────────┐
                                       上一幀 hidden │   LSTM (1~2 層)       │ → 更新 hidden state
                                       state ───────→│  「腦中的筆記本」     │   傳給下一幀
                                                    └──────────────────────┘
                                                                  │
                                                                  ▼
                                                     action head (MSE 回歸)
                                                                  │
                                                                  ▼
                                                    8 維動作 (7 joints + 1 gripper)
```

**設計選擇（第一版）：**
- **Image encoder**：ResNet18（LeRobot 內建有 backbone 可重用），影像降到 **96×96** 省記憶體。
- **Action head**：**簡單 MSE 回歸**（直接吐 8 個數字）。理由：FSM demo 乾淨一致、掀杯動作單純，
  不需要 Robomimic 的 GMM head。之後若動作端不穩再考慮升級 GMM。
- **不做 action chunking**：每步輸出單一動作，最符合 recurrent 的逐步推論。

> 進階備案（先不做，列在這供日後對照）：LSTM-GMM head（完全對齊 Robomimic）、
> action chunking（對齊 eval 的 `action_horizon` 機制）。

---

## 3. 最核心的技術難點：訓練時的 BPTT 記憶體

LSTM 的訓練要做 **BPTT（Backpropagation Through Time）**：把「錯多少」的訊號
沿時間軸從 Act 階段一路傳回 Reveal 階段。要傳回 N 幀，就得同時把 N 幀的影像與計算結果
存在 GPU 記憶體裡 → 幀數越多越吃記憶體。

**我們的序列長度（已從實際 dataset metadata 量測，非估計）：**

| 難度 | 平均 episode 幀數 | episodes |
|------|------|------|
| 0 次洗牌 | **514** | 99 |
| 2 次洗牌（預設）| **574** | 99 |
| 5 次洗牌 | **664** | 99 |

- fps=30；每多一次洗牌 +30 幀（= `shuffle_per_swap`），**證實 Phase 1–3 逐幀有錄**。
- episode 比原估長很多（17–22 秒）→ **記憶要跨越 500+ 幀**，遠超連續視窗能涵蓋的範圍。

**關鍵推論：訓練視窗必須涵蓋「Reveal→Act」整段，否則 LSTM 學不到記憶。**
若訓練視窗只取 episode 中段（沒含 reveal 那 ~50 幀），LSTM 從零 hidden state 開始、
看不到球的初始位置 → 無法對應到 demo 的正確杯子 → 只會學到「平均軌跡」。
所以**不能用任意子視窗，必須 strided 涵蓋整段 episode**。
→ 用 config 的 delta_indices 設成「跨整段 + 跳幀」，例如 episode~520 幀：
  `stride=8, L≈66`（66×8=528）就能涵蓋整段，且 BPTT 只回傳 66 步。

> episode 長度隨難度變動（514→664），但 config 的 delta_indices 是固定的。
> 作法：針對該難度用「能蓋住最長 episode」的 (L, stride)，較短 episode 由 lerobot 自動 padding + mask。
> LSTM forward 要尊重 pad mask（padding 幀不算 loss）。

直接硬做整段連續 BPTT + 兩顆相機影像，在 4090 24GB 上會爆。四個壓記憶體的槓桿
（按「最該先用」排序）：

1. **Strided 跳幀（最對症）**：洗牌很慢，不需每幀都看。整段 episode 每 3 幀取 1 幀，
   270 幀 → ~90 幀，**仍涵蓋完整 Reveal→Act 跨度**，BPTT 長度砍成 1/3。
   （= survey §6.4 的 strided history。）
2. **降影像解析度（記憶體最大槓桿）**：96×96 或 128×128，洗牌的粗略空間資訊足夠。
3. **小 batch + gradient accumulation**：batch 4~8，必要時梯度累加補回等效 batch。
4. **Gradient checkpointing（進階備案）**：用重算換記憶體，要更長序列時才動用。

**結論**：ResNet18 + 96px + 跳幀到 ~90 幀 + batch 8 → 整段 BPTT 在 4090 上很輕鬆，
連 5 次洗牌都壓得住。

---

## 4. 與既有 codebase 的整合點（已實際看過 code）

ShellBench 的 env / datagen / eval / sweep 都已完成。LSTM 只需「插進去」，不改既有流程。

### 4.1 評估端 — 幾乎免費 ✅
`scripts/eval_shell_game.py` 已經 `import simulator.tasks`，並用
`get_policy_class(policy_type)` + `from_pretrained()` 載入 policy。
只要我們的 LSTM policy 在 **simulator 套件的 import chain** 裡完成註冊，eval 端就自動認得。

**重要 caveat（recurrent policy 專屬）**：
`eval_shell_game.py` 的 `EvalPolicy.observe()` 在 Phase 1–3 每步呼叫 `select_action()`
餵 observation，然後呼叫 `_clear_cached_actions()`。
→ 對 LSTM 來說，Phase 1–3 餵 observation 正好讓 hidden state 累積 Reveal/Shuffle 資訊（完美）。
→ **但 `_clear_cached_actions()` 絕不能清掉 hidden state**，只能清 action queue。
   我們的 policy 要確保 `select_action()` 內部維護 hidden state，且不要把它做成
   `_clear_cached_actions()` 會掃到的 `action_queue` 之類欄位。
→ `policy.reset()`（每 episode 開頭呼叫）才是清 hidden state 的地方。

### 4.2 訓練端 — 需要一個 thin wrapper
`lerobot-train` 是獨立 CLI，**不會** import 我們的 simulator 套件，所以光靠註冊還不夠。
`scripts/run_sweep.py` 的 `run_training()` 支援 `training.backend: command`
（跑使用者自訂指令模板）。→ 乾淨的做法：

```
LSTM/scripts/train_lstm.py   # thin wrapper:
                             #   1. import 我們的 LSTM policy（觸發註冊）
                             #   2. 呼叫 lerobot 的 train main（或 subprocess lerobot-train）
```

然後在 LSTM 的 sweep config 設 `training.backend: command`、`training.command: "<python> LSTM/scripts/train_lstm.py ..."`。
**完全不用 fork lerobot。**

### 4.3 Sweep config
複製既有 `configs/experiments/exp1_shuffle_scaling/sweep.yaml` 的形式，
在 `LSTM/configs/` 放 LSTM 專用 sweep（`policy.type: lstm`、`training.backend: command` 等）。
Dataset 與既有 ACT/DP **共用同一份**（同 `repo_id`），只換 policy，比較成本低。

---

## 5. LeRobot 整合機制（已讀 Docker 內 source 確認，lerobot 0.4.2）

> 已直接讀過 Docker image 裡的 `lerobot/policies/factory.py`、`configs/policies.py`、
> `scripts/lerobot_train.py`，以下為**確認事實**，非假設。

**實際版本：lerobot 0.4.2**（`pyproject.toml` 的 pin 為準；`uv.lock` 雖寫 0.4.4，Docker image 裝的是 0.4.2）。

**機制（關鍵）：**
- `PreTrainedConfig` 是 `draccus.ChoiceRegistry`，built-in 用 `@PreTrainedConfig.register_subclass("act")` 註冊。
  → 我們 `@PreTrainedConfig.register_subclass("lstm")` 後，`--policy.type=lstm` **就能被 parse**。
- **但** `get_policy_class()`、`make_policy_config()`、`make_pre_post_processors()`
  **全部是寫死的 if/elif**，沒有 plugin/registry。它們不認得 "lstm"。
- `lerobot_train.py` 的訓練迴圈：
  `make_policy()`（→ `get_policy_class(cfg.type)`）→ `make_pre_post_processors()` →
  迴圈內 `loss, out = policy.forward(batch)`。

**→ 整合策略（不 fork lerobot）：** 寫一個 `LSTM/scripts/train_lstm.py` wrapper：
1. `import` 我們的 LSTM 模組（觸發 `register_subclass`，讓 `--policy.type=lstm` 可解析）
2. **monkeypatch** `lerobot.policies.factory.get_policy_class` 與 `make_pre_post_processors`，
   讓它們在遇到 lstm / 我們的 config 時回傳我們的 class / processor
3. 呼叫 `lerobot.scripts.lerobot_train.main()`

eval 端（`eval_shell_game.py`）同理：它也用 `get_policy_class`，所以 eval 也要先 import 我們的模組
並套同樣的 patch（可包成一個 `LSTM/policy/register.py`，import 即完成註冊 + patch）。

**BPTT 序列取樣 = config 的 delta_indices（重要，省掉客製 dataset）：**
lerobot 用 config 的 `observation_delta_indices` / `action_delta_indices`（回傳相對幀索引 list）
自動建 `delta_timestamps`，dataset 就會把每個 sample 變成一段「長度 L 的時間窗」。
- 連續 L 幀：`list(range(1 - L, 1))`
- **strided（跳幀）**：`list(range(1 - L*stride, 1, stride))` ← §3 的跳幀直接這樣表達！
所以訓練時每個 batch sample 形狀就是 `[B, L, ...]`，LSTM 對這 L 步做 forward + BPTT，
**完全不用自己寫 dataset sampling**。

### 開工前仍要抽查的兩件事
1. **Dataset 有錄 Phase 1–3**：抽一筆確認 Reveal/Shuffle 的幀真的在 dataset 裡（survey Risk 4）。
   若沒錄，任何 memory policy 都學不到東西。
2. **實際 episode 幀數 / fps**：dataset 標 30 fps、sim step 60 Hz；確認一筆 episode 實際幾幀，校準 stride。

---

## 6. 分階段執行計畫（先打通管線，再追記憶）

LSTM 路線有兩類風險疊在一起：**整合風險**（plugin 註冊、訓練/eval 接線）與
**記憶學習風險**（BPTT 學不學得到長依賴）。策略是先清掉整合風險。

```
Phase 0 — 驗證環境（§5）
  → 確認 lerobot 版本、註冊機制、dataset 有錄 Phase 1–3、實際幀數

Phase 1 — 寫 LSTM policy plugin（config + policy class），MSE head
  → 驗證：能 import 且 get_policy_class("lstm") 拿得到 class

Phase 2 — train wrapper + 簡單難度打通整條管線
  → num_shuffles = 0 或 1（episode 短，BPTT 便宜）
  → 驗證：lerobot-train 跑得起來 → eval 出 metrics.json → DSR 明顯 > 隨機線
  → 這一步證明「整合」沒問題，先不追記憶能力

Phase 3 — 加大難度 + 調記憶旋鈕（strided / 解析度 / 序列長度）
  → num_shuffles 掃 0..5，畫出 DSR 衰減曲線
  → 這才是 benchmark 要的主結果

Phase 4 —（選做）對齊 Robomimic：升級 GMM head / 調 LSTM 層數
```

---

## 7. 第一版預定檔案

```
Advanced-level/LSTM/
├── docs/
│   └── implementation_plan.md        # 本文件
├── policy/
│   ├── configuration_lstm.py         # PreTrainedConfig 子類（超參數 + 註冊）
│   ├── modeling_lstm.py              # PreTrainedPolicy 子類（encoder + LSTM + head）
│   └── __init__.py                   # import 即完成註冊
├── scripts/
│   └── train_lstm.py                 # thin wrapper：註冊後呼叫 lerobot train
└── configs/
    └── exp_lstm_shuffle_scaling/
        └── sweep.yaml                # LSTM 專用 sweep（共用既有 dataset）
```

---

## 7.1 Plugin 確切 API（已讀 lerobot 0.4.2 abstract 介面）

**`configuration_lstm.py` — `@PreTrainedConfig.register_subclass("lstm")` 的 dataclass，必須實作：**
- `observation_delta_indices` (property) → 例如 `range(1 - L*stride, 1, stride)`（BPTT 窗 + 跳幀）
- `action_delta_indices` (property) → 與 observation 對齊的 list
- `reward_delta_indices` (property) → `None`（IL 不用）
- `get_optimizer_preset()` → 回傳 `OptimizerConfig`（如 AdamConfig）
- `get_scheduler_preset()` → 回傳 `LRSchedulerConfig | None`
- `validate_features()` → 檢查有影像/狀態/action feature
- 自訂超參數欄位：`hidden_size`、`num_lstm_layers`、`seq_len (L)`、`stride`、
  `image_size`、`vision_backbone` 等。

**`modeling_lstm.py` — `PreTrainedPolicy` 子類，必須有：**
- class 屬性 `config_class = LSTMConfig`、`name = "lstm"`
- `get_optim_params()` → 回傳 optimizer 參數
- `reset()` → **清 hidden state**（每 episode 開頭。注意：只清 hidden state）
- `forward(batch) -> (loss, dict)` → 對 `[B, L, ...]` 序列做 BPTT，回傳 MSE loss
- `select_action(batch) -> action` → 推論單步；**內部維護 hidden state**，
  不要做成會被 eval `_clear_cached_actions()` 掃到的 `action_queue` 欄位

**`register.py` — import 即完成註冊 + patch：**
- import config/model（觸發 `register_subclass`）
- monkeypatch `lerobot.policies.factory.get_policy_class`（+ 我們的 lstm 分支）
- 提供 `make_lstm_pre_post_processors()` 並 patch `make_pre_post_processors`
  （normalization 用既有 helper，仿照 `processor_act.py`）

> 已確認 batch 進 `forward` 前會先過 `preprocessor(batch)`（normalization）。

---

## 7.5 訓練 Protocol（主結果用哪種，**必須跟組員對齊**）

記憶曲線（DSR vs num_shuffles / num_cups）有兩種訓練法，意義不同：

**Protocol A — 每難度各訓一個 policy（matched train/eval）**
- `ns=k` 的 dataset → 訓一個 policy → 在 `ns=k` 評估 → 曲線上一個點。
- 每個難度的 dataset 本身就含該難度的洗牌過程，所以 policy 會學到對應的記憶負荷
  （不是「只學最簡單」—— 那只是 ns=0 de-risk run 的假象）。
- **這是主結果**：因為 LSTM 要跟組員的 ACT / Diffusion / VQ-BeT / Oracle 疊在同一張圖比較，
  大家**必須用同一套 protocol**，曲線才可比。→ **開工 Phase 3 前先跟組員確認大家都用 A。**

**Protocol B — 混合所有難度訓一個 generalist（額外 ablation）**
- ns=0..5 dataset 全混在一起訓**一個** policy，再分別在各難度評估。
- 測「一個 agent 同時應付各種記憶負荷」，較接近真實；lerobot 支援多 dataset 合併。
- 實務好處：每難度只有 ~99 筆 demo，對 LSTM 學長期記憶偏少；混起來 ~600 筆，
  高難度的點可能學得更好。
- **定位**：加分的延伸實驗，**分開報告**，不可拿去取代 A（否則跟其他 policy 不可比）。

> 結論：**A 必做（主菜，須與組員同 protocol）；B 建議加做（generalist vs specialist）。**

---

## 8. 待決定事項（跟組員/老師對齊用）

- [ ] **訓練 protocol（最重要）**：主結果用 Protocol A（每難度各訓，matched）— 確認組員的
      ACT/DP/VQ-BeT 也用同一套；是否加做 Protocol B（混合 generalist）當 ablation。見 §7.5。

- [ ] Action head：MSE（建議先做） vs. GMM（對齊 Robomimic）
- [ ] BPTT 序列：整段+跳幀（建議） vs. truncated 短序列
- [ ] LSTM 層數 / hidden size（先用 2 層 / 512 之類保守值）
- [ ] 影像解析度（先 96×96）
- [ ] 是否同時用 front+wrist 兩相機，還是只用 front（記憶主來源）

> 標「建議」者為第一版預設選擇；除非有理由，否則照建議走，先把曲線跑出來再優化。
