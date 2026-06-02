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
| `configs/` | LSTM 專用 sweep 設定（Phase 3 用） |

## 現況

- [x] 專案架構與文件理解
- [x] LSTM 機制與整合策略討論
- [x] Implementation plan（`docs/implementation_plan.md`）
- [x] Phase 0：驗證環境
  - [x] lerobot 版本（**0.4.2**）+ 註冊機制（讀 Docker source 確認，見 plan §5）
  - [x] dataset 抽查通過：洗牌每+1=+30幀 → Phase 1–3 逐幀有錄 ✅；
        fps=30；episode 514–664 幀；2 cam 480×640；state=9 action=8
  - [x] 可共用組員既有 dataset（`johnnyli1220/shellbench-num_shuffles-{0..5}` 等，public）
- [x] Phase 1：LSTM policy plugin（config + model + processor + register/patch）
  - [x] 容器內 smoke test 通過：`python LSTM/scripts/smoke_test.py`（註冊 + BPTT + 推論記憶）
- [~] Phase 2：train/eval wrapper + 執行手冊
  - [x] `train_lstm.py`（容器內 `--help` 確認 lstm 已是 policy 選項、超參數變成 flag）
  - [x] `eval_lstm.py`（零侵入跑 eval_shell_game）
  - [x] `docs/running.md` 執行手冊
  - [ ] 在容器內用 num_shuffles=0 dataset 實際訓練 + 評估，確認 DSR > 1/3
- [ ] Phase 3：加大難度（num_shuffles 0..5），畫記憶曲線

## 背景文件（repo 既有）

- `documents/training_policy_survey.md` — policy 選型調查（LSTM 在 §6.1）
- `advanced_docs/task_description.md` — ShellBench 任務定義
- `advanced_docs/IMPLEMENT_RECORD.md` — 既有 task/datagen/eval 實作紀錄
- `scripts/eval_shell_game.py`, `scripts/run_sweep.py` — 整合接線點
