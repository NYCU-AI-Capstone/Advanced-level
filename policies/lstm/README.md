# ShellBench — LSTM Policy

ShellBench 的 **recurrent (LSTM) imitation-learning policy** 開發資料夾。
定位：作為「真的有跨整段 episode memory」的對照組，跟 built-in 的
ACT / Diffusion / VQ-BeT 比較記憶能力曲線。

## 資料夾結構

| 路徑 | 用途 |
|------|------|
| `docs/` | 設計與實作文件 |
| `docs/how_it_works.md` | **組員先讀這份** — 白話講流程、檔案各管什麼、怎麼跑 |
| `docs/running.md` | **要實際跑就看這份** — 訓練 / 評估指令、旋鈕、OOM 對策、判讀 |
| `docs/implementation_plan.md` | 技術藍圖 — 架構、整合點、BPTT/GPU 取捨、分階段計畫 |
| `policy/` | LSTM policy plugin（config / model / processor / register） |
| `scripts/smoke_test.py` | plugin 煙霧測試（註冊 + 訓練 + 推論） |
| `scripts/train_lstm.py` | 訓練 wrapper（註冊後呼叫 lerobot-train） |
| `scripts/eval_lstm.py` | 評估 wrapper（註冊後跑 eval_shell_game，零侵入） |
| `scripts/decode_dataset_to_images.py` | 把 AV1 影片 dataset 解碼成圖片（加速訓練） |
| `scripts/diagnose_rollout.py` | 離線診斷（預測 vs 真實動作 MSE） |
| `configs/` | LSTM 專用 sweep 設定（Phase 3 用） |

## 現況（2026-06-05）

- [x] Phase 0：環境驗證（lerobot 0.4.2、dataset 確認）
- [x] Phase 1：LSTM policy plugin（config + model + processor + register + smoke test）
- [x] Phase 2：train/eval wrapper + 執行手冊
- [~] Phase 2.5：實際訓練 + 評估，調整架構
  - [x] 96px / ns=3 / 100k steps → DSR 0.25（比隨機差，失敗）
  - [x] 128px / ns=3 / 80k steps → DSR 0.40（略高於隨機 0.33）
  - [x] 160px / ns=0 / 50k steps → DSR 0.55（模型永遠選 cup 0）
  - [x] 認為的可能問題：**MSE mode averaging**（模型學到平均軌跡，不區分杯子）
  - [x] 解法：加 **cup classification head**（已實作，尚未訓練驗證）
- [ ] Phase 3：掃 num_shuffles 0..5，畫記憶衰減曲線

## 關鍵發現與待解問題

### 已確認
1. **MSR = 1.0**：操作能力（掀杯子）完全沒問題，每次都成功
2. **image_size 96px 太小**：球只有 1-2 pixel，模型看不到 → DSR 比隨機低
3. **128px 有改善但不夠**：DSR 從 0.25 提升到 0.40
4. **160px 可能是最好的**

### 根本問題：MSE mode averaging
模型永遠選同一個杯子（cup 0），DSR 高只是因為碰巧球在那。原因：
- Dataset 裡 1/3 demo 去 cup 0、1/3 去 cup 1、1/3 去 cup 2
- MSE loss 把三種方向的軌跡取平均 → 模型學到「去平均位置」
- LSTM hidden state 即使記住球位置，MSE 也不獎勵它區分杯子

### 解法：Cup Classification Head（已實作，待訓練驗證）
在 action head 旁加一個 3-way classification head：
- 從 LSTM hidden state 預測「球在哪個杯子」
- Cross-entropy loss 強制 hidden state 編碼離散的杯子選擇
- Total loss = MSE + cup_loss_weight × CE
- Target cup label 從 ground truth action 的 base joint (dim 0) 自動推斷

詳見 `docs/experiment_log.md`。

## 已上傳的模型

| HuggingFace Repo | 設定 | DSR |
|---|---|---|
| `tsukimiyo0202/shellbench-lstm-ns3-128px` | ns=3, 128px, 80k steps | 0.40 |

## 背景文件（repo 既有）

- `documents/training_policy_survey.md` — policy 選型調查（LSTM 在 §6.1）
- `docs/shellbench/task_description.md` — ShellBench 任務定義
- `docs/shellbench/IMPLEMENT_RECORD.md` — 既有 task/datagen/eval 實作紀錄
- `scripts/eval_shell_game.py`, `scripts/run_sweep.py` — 整合接線點
