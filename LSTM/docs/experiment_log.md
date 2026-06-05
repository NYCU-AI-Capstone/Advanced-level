# LSTM Policy 實驗紀錄

> 按時間順序記錄所有訓練 / 評估嘗試、發現的問題、以及架構改動。
> 接手的人先看這份，再看 `how_it_works.md`（架構）和 `running.md`（指令）。

---

## 實驗總覽

| # | 日期 | image_size | num_shuffles | steps | batch | DSR | MSR | κ | 備註 |
|---|------|-----------|-------------|-------|-------|-----|-----|---|------|
| 1 | 06-03 | 96 | 3 | 100k | 2 | **0.25** | ? | ? | 比隨機(0.33)還低；96px 看不到球 |
| 2 | 06-04 | 128 | 3 | 80k | 8 | **0.40** | 1.0 | 0.10 | 略高於隨機；模型偏好 cup 0 和 cup 2 |
| 3 | 06-05 | 160 | 0 | 50k | 8 | **0.55** | 1.0 | 0.33 | 20 次全選 cup 0；DSR 高只因球碰巧在 cup 0 |

---

## 實驗 1：96px / ns=3 / 100k steps

**設定：**
- `image_size=96, seq_len=200, obs_stride=3, batch_size=2`
- Dataset: `johnnyli1220/shellbench-num_shuffles-3`
- 100k steps（約 28 小時）
- 平台：GlowsAI RTX 4090

**結果：** DSR = 0.25（比隨機 0.33 還低）

**分析：**
- 96px 下球只有 1-2 pixel，跟杯子底部的高光幾乎無法區分
- 模型可能學到了「錯誤的對應」——把模糊亮點跟隔壁杯子關聯
- 決定提高 image_size

---

## 實驗 2：128px / ns=3 / 80k steps

**設定：**
- `image_size=128, seq_len=200, obs_stride=3, batch_size=8`
- Dataset: 同上（decode 成 160px image 版，模型內部 resize 到 128）
- 80k steps
- GPU 用量：~21 GB / 24 GB（87%）

**結果：** DSR = 0.40, MSR = 1.0, κ = 0.10

**逐集分析：**
```
selected=0: 7 次    selected=1: 2 次    selected=2: 8 次（偏好 cup 0 和 2，避開 cup 1）
truth=0 時正確率: 4/5 (80%)
truth=1 時正確率: 1/4 (25%)
truth=2 時正確率: 3/7 (43%)
```

**分析：**
- MSR = 1.0 代表操作能力完美（每次都成功掀起杯子）
- DSR 0.40 > 0.33 但可能是位置偏好碰巧吻合，不是真正追蹤球
- 模型有強烈的 cup 0 / cup 2 偏好
- 模型已上傳：`tsukimiyo0202/shellbench-lstm-ns3-128px`

---

## 實驗 3：160px / ns=0 / 50k steps

**目的：** 用最簡單的難度（ns=0，球不動）驗證 LSTM 有沒有學到記憶

**設定：**
- `image_size=160, seq_len=200, obs_stride=3, batch_size=8`
- Dataset: `johnnyli1220/shellbench-num_shuffles-0`（decode 成 160px image 版）
- 50k steps
- GPU 用量：~23.6 GB / 24 GB（96%，很緊但沒 OOM）

**結果：** DSR = 0.55, MSR = 1.0, κ = 0.33

**逐集分析（關鍵發現）：**
```
20 次評估，模型「全部」選 cup 0：
  truth=0: 11 次 → 全部 CORRECT（11/11）
  truth=1:  2 次 → 全部 WRONG
  truth=2:  7 次 → 全部 WRONG
DSR = 11/20 = 0.55，完全是因為 truth=0 出現了 11 次
```

**結論：模型完全沒有學到記憶，永遠去 cup 0。**

---

## 根本問題診斷：MSE Mode Averaging

### 問題
MSE (Mean Squared Error) 在面對多模態動作分布時會「取平均」：
- 訓練集裡 ~1/3 的 demo 去 cup 0（左）、~1/3 去 cup 1（中）、~1/3 去 cup 2（右）
- 三種方向的軌跡做 MSE → 梯度拉向「平均位置」
- LSTM hidden state 即使記住了球在哪，MSE loss 也不獎勵它根據記憶「選不同杯子」
- 結果：模型學到一條固定的「平均軌跡」，恰好最靠近 cup 0

### 佐證
1. 實驗 3 的 20 次評估全選 cup 0（沒有任何條件性行為）
2. 實驗 2 強烈偏好 cup 0 和 cup 2，幾乎不選 cup 1
3. 這個問題跟 image_size 和 num_shuffles 無關——是 loss function 層級的問題

### 驗證方法
用 k-means 對 dataset 的 action dim 0（base joint）做聚類，確認三個杯子對應的動作模式：
```
Cluster 0: a0 ≈ -0.07（左，34 episodes）
Cluster 1: a0 ≈ +0.08（右，38 episodes）
Cluster 2: a0 ≈  0.00（中，27 episodes）
三群完全不重疊，分得非常乾淨。
```

---

## 解法：Cup Classification Head（已實作，待訓練驗證）

### 原理
在既有的 MSE action head 旁邊，加一個 **3-way classification head**：
- 輸入：LSTM hidden state（序列後 30% 的平均）
- 輸出：3 個 logits（cup 0 / 1 / 2）
- Loss：Cross-entropy
- Total loss = MSE loss + `cup_loss_weight` × CE loss

CE loss 提供「離散的梯度信號」，強制 LSTM hidden state 編碼「球在哪個杯子」。
MSE 負責學「怎麼掀杯子」（操作技能），CE 負責學「掀哪個杯子」（記憶決策）。

### Target Cup Label（自動推斷，不需改 dataset）
從 ground truth action 的 base joint (dim 0) 推斷 target cup：
- 取序列後 30% 的 action dim 0 平均值（normalized space）
- `< -0.7` → cup 0（左）
- `> +0.7` → cup 2（右）
- 其餘 → cup 1（中）

閾值在 normalized space（MEAN_STD normalization），三群中心分別在 -1.5 / -0.06 / +1.5，
間距足夠大，不會誤判。

### 改動的檔案
- `policy/configuration_lstm.py`：新增 `num_cups=3`, `cup_loss_weight=1.0`
- `policy/modeling_lstm.py`：
  - 新增 `cup_head`（Linear → ReLU → Linear → 3 classes）
  - 新增 `_derive_cup_target()` 靜態方法
  - `forward()` 改為 MSE + CE 混合 loss
  - `select_action()` 順便算 `_cup_logits`（推論時可查看模型的杯子預測）

### 訓練指令
```bash
python LSTM/scripts/train_lstm.py \
  --policy.type=lstm \
  --dataset.repo_id=johnnyli1220/shellbench-num_shuffles-0 \
  --dataset.root=/workspace/aicapstone/data/lerobot_img/johnnyli1220/shellbench-num_shuffles-0 \
  --policy.device=cuda \
  --policy.seq_len=200 \
  --policy.obs_stride=3 \
  --policy.image_size=160 \
  --policy.cup_loss_weight=1.0 \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=50000 \
  --num_workers=4 \
  --save_freq=10000 \
  --log_freq=500 \
  --wandb.enable=false \
  --output_dir=LSTM/outputs/ns0_160px_cup_v1 \
  --job_name=lstm_ns0_160px_cup
```

### 預期結果
- 如果 cup classification head 有效：
  - `cup_loss` 應該快速下降（< 0.5 = 模型能區分三個杯子）
  - DSR 應該明顯 > 0.55，且不再永遠選同一個杯子
  - ns=0 的 DSR 應該接近 1.0（球不動，只要記住就好）
- 如果 DSR 還是差：
  - 檢查 `cup_loss` 有沒有下降（如果沒有 → label 推斷有問題）
  - 嘗試加大 `cup_loss_weight`（如 2.0 或 5.0）
  - 考慮換 GMM head（更根本地解決多模態問題）

---

## 下一步（給接手的人）

### 最優先：驗證 cup classification head
1. 用上面的指令跑 ns=0 + cup head 訓練
2. 觀察 log 裡的 `cup_loss` 是否下降
3. 跑 eval，看 DSR 是否改善、是否不再永遠選同一個杯子

### 驗證通過後：掃曲線（Phase 3）
1. 用 Protocol A（每個難度各訓一個 model）
2. ns = 0, 1, 2, 3, 4, 5 各跑一次 train + eval
3. 畫 DSR vs num_shuffles 曲線
4. 跟組員的 ACT / Diffusion / VQ-BeT / Oracle 疊在一起

### Dataset 準備狀態
| Dataset | 已下載 (host) | 已 decode 160px |
|---------|:---:|:---:|
| ns=0 | ✅ | ✅ |
| ns=1 | ❌ | ❌ |
| ns=2 | ❌ | ❌ |
| ns=3 | ✅ | ✅ |
| ns=4 | ❌ | ❌ |
| ns=5 | ❌ | ❌ |

下載指令：見 `running.md` §0.5。
Decode 指令：見 `running.md` §⚡ 加速。

### 重要注意事項
- 容器裡用 `python`（3.11），host 用 `python3`（3.12）
- 容器是 `--rm`，cache 重開就沒 → dataset 要存在 `/workspace/aicapstone/data/` 底下
- 訓練長時間跑要用 `tmux`（容器內 `apt-get install -y tmux`）
- GPU 記憶體：160px + batch=8 用 96% (23.6 GB)，很緊；如果 OOM 降 batch 到 4
- 評估前先停訓練，兩個搶 GPU
