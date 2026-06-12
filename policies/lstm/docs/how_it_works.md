# LSTM Policy 怎麼運作（給組員的說明）

這份文件用白話講「我們做了什麼、檔案各管什麼、整個流程怎麼跑」。
技術細節與設計取捨在 [`implementation_plan.md`](./implementation_plan.md)。
實驗紀錄與目前進度在 [`experiment_log.md`](./experiment_log.md)。

---

## 一句話總結

我們幫 ShellBench 加了一個 **LSTM（有記憶的）policy**，當作「真的能記住整段洗牌過程」的對照組，
跟 LeRobot 內建、只看最近幾幀的 ACT / Diffusion 比較。
LSTM 不是 LeRobot 內建的，所以我們自己寫了一個 plugin 接進去。

---

## 為什麼要 LSTM？

ShellBench 是「猜杯子」：球藏在哪只在開頭 (Reveal) 看得到，之後蓋住、洗牌，
機器人要到最後 (Act) 才掀杯子，中間隔了 **500 多幀**。

- **ACT / Diffusion**：只看最近 1~2 幀 → 球早就被蓋住了，等於用猜的。
- **LSTM**：有一個「腦中筆記本」(hidden state)，把開頭看到的球位置一路記到最後。

所以 LSTM 在洗牌變多時，DSR（選對率）應該掉得比 ACT/Diffusion 慢 —— 這個對比就是我們要的結果。

---

## 檔案地圖（`policies/lstm/policy/`）

```
policies/lstm/policy/
├── configuration_lstm.py   ← 超參數 + 「要取哪幾幀來訓練」
├── modeling_lstm.py        ← 模型本體（ResNet18 + LSTM + 動作輸出 + 杯子分類）
├── processor_lstm.py       ← 影像/數值的正規化前後處理
├── register.py             ← 把以上接進 LeRobot 的「膠水」
└── __init__.py             ← import 入口
```

| 檔案 | 管什麼 | 白話 |
|------|--------|------|
| `configuration_lstm.py` | 設定 | LSTM 有幾層、影像縮多小、**訓練時一個樣本涵蓋幾幀**。讓 `--policy.type=lstm` 能被認得。 |
| `modeling_lstm.py` | 模型 | 影像→ResNet18 壓成特徵→LSTM 攜帶記憶→輸出 8 維動作 + 3-way 杯子分類。訓練走 `forward`，跑機器人走 `select_action`。 |
| `processor_lstm.py` | 前後處理 | 進模型前把影像/數值正規化，出模型後還原。直接沿用 LeRobot 內建零件。 |
| `register.py` | 接線 | LeRobot 的 policy 清單是寫死的，認不得 lstm。這裡用 monkeypatch 把 lstm 塞進去，**不用改 LeRobot 原始碼**。 |

---

## 它怎麼接進 LeRobot（重點）

LeRobot 內建 policy（act/diffusion/...）是**寫死在程式裡**的清單，沒有外掛機制。
我們用兩招把 LSTM 塞進去，**完全不改 LeRobot 的程式碼**：

1. `@PreTrainedConfig.register_subclass("lstm")` → 讓 `--policy.type=lstm` 這個指令能被解析。
2. `register.py` 的 monkeypatch → 讓 LeRobot 內部「拿 policy 類別」「建前後處理」的函式認得 lstm。

**所以規則很簡單：任何要用到 lstm 的程式，開頭先 `import policies.lstm.policy.register` 就好。**

---

## 整個流程怎麼跑

```
①  Dataset（組員已生好，放在 HuggingFace）
    johnnyli1220/shellbench-num_shuffles-{0..5} 等，含 front+wrist 影像、關節、動作
            │
            ▼
②  訓練  lerobot-train --policy.type=lstm ...
    每個訓練樣本 = 一段「長度 L 的影像/動作序列」（由 config 的 delta_indices 決定）
    LSTM 對這段序列做 BPTT，學會「把開頭的球位置記到最後再選杯子」
    Loss = MSE（動作預測）+ CE（杯子分類）
            │
            ▼
③  產出 checkpoint（模型權重 + config）
            │
            ▼
④  評估  eval_shell_game.py（在 Isaac Sim 裡跑）
    - Phase 1-3（洗牌）：每幀餵進 LSTM，hidden state 累積記憶（不動機器人）
    - Phase 4（掀杯）：LSTM 根據記憶選一個杯子去掀
    - 算出 DSR / MSR / SR / κ
```

② 和 ④ 都在 **Docker 容器內**跑（lerobot + Isaac Sim 都在那）。

---

## 幾個關鍵設計（為什麼這樣做）

1. **動作輸出用 MSE 回歸 + cup classification head。**
   MSE 學「怎麼掀杯子」（操作技能），CE 學「掀哪個杯子」（記憶決策）。
   純 MSE 會把三個方向的軌跡取平均（mode averaging），導致模型永遠去同一個杯子。
   Cup classification head 用 cross-entropy 提供離散信號，強制 LSTM 區分三個杯子。
   詳見 `experiment_log.md`。

2. **訓練序列用「跳幀涵蓋整段」**。
   episode 有 500+ 幀，全部硬塞 GPU 會爆。用 `seq_len`(L) + `obs_stride` 控制要取幾幀、隔多遠。
   目前用 `seq_len=200, obs_stride=3` → 涵蓋 600 幀，蓋住整段 episode（Reveal → Act）。

3. **評估時 hidden state 跨整段 episode 攜帶**。
   `select_action` 每被呼叫一次就更新一次記憶；`reset()` 在每個 episode 開頭清空。
   evaluation 腳本在洗牌階段也會每幀餵觀測進來，正好讓 LSTM 把洗牌過程記下來。

4. **先在最簡單難度（num_shuffles=0）打通**，再加難度。
   先確認「plugin 能註冊、訓練跑得起來、評估出得了數字、DSR 高於亂猜」，把整合問題清掉，
   再去掃洗牌次數畫記憶曲線。

---

## 怎麼確認 plugin 沒壞（在容器內）

```bash
cd /workspace/aicapstone
python policies/lstm/scripts/smoke_test.py
```

會驗證三件事（不需 dataset / GPU）：①lerobot 認得 `--policy.type=lstm` ②訓練 forward+BPTT 跑得動
③推論 select_action 攜帶記憶。看到 `✅ ALL SMOKE TESTS PASSED` 就 OK。

## 目前狀態

- ✅ plugin 五個檔案（config / model / processor / register / init）
- ✅ 容器內 smoke test 通過
- ✅ train/eval wrapper 完成
- ✅ 已跑三輪實驗（96px/128px/160px），確認操作能力 OK (MSR=1.0)
- ✅ 發現 MSE mode averaging 問題，已實作 cup classification head
- ⬜ **下一步：用 cup classification head 重新訓練，驗證 DSR 改善**
- ⬜ 掃 num_shuffles 0..5 畫記憶曲線（Phase 3）

> 詳細實驗紀錄和下一步指引見 [`experiment_log.md`](./experiment_log.md)。
