# LSTM 實驗與調整紀錄

## 1. 目標

ShellBench 的球只在 Reveal 階段可見，之後會被杯子蓋住並經過多次洗牌，直到 Act
階段才由機器人選杯。因此模型除了需要完成抓取動作，也必須保留跨越數百幀的記憶。

本實驗的目標是建立一個可整合至 LeRobot 的 LSTM policy，並找出在記憶能力、操作穩定性、
訓練成本和 GPU 記憶體之間表現較好的設定。

評估時主要觀察：

- **DSR (Decision Success Rate)**：是否選到藏球的正確杯子，主要反映記憶與追蹤能力。
- **MSR (Manipulation Success Rate)**：是否成功掀起任一杯子，主要反映操作能力。
- **SR (Success Rate)**：同時選對杯子並成功完成操作。
- **Cohen's kappa**：扣除三選一隨機猜測後的 DSR。

---

## 2. 完成的模型與訓練管線調整

除了超參數實驗之外，也完成了以下實作與工程調整：

1. 實作 LeRobot LSTM policy plugin，包括 config、model、processor 與 policy registration。
2. 使用 ResNet 擷取 front 和 wrist 相機特徵，再與 robot state 串接後送入 LSTM。
3. 在每個 episode 之間重設 hidden state，並在 Reveal、Cover、Shuffle 階段持續更新記憶。
4. 讓訓練與評估共用相同的 `obs_stride`，避免時間尺度不一致。
5. 加入 action chunking，使模型可一次預測多個未來動作。
6. 加入 `current_obs_frames`，允許 action head 直接取得近期影像特徵，不必完全依賴 hidden state。
7. 加入 gradient checkpointing，以支援長序列 BPTT。
8. 將 visual backbone 與其他模型參數拆成不同 optimizer parameter groups，可分別設定 learning rate。
9. 修正 LSTM 權重儲存至 safetensors 時的 shared-storage 問題。
10. 加入 checkpoint 自動 evaluation，依 DSR 保存最佳 checkpoint。
11. 將 AV1 dataset 預先解碼成 image dataset，降低訓練期間的 CPU 解碼瓶頸。
12. 在 evaluation 加入 `max_act_steps`，避免失敗 episode 一直執行到完整 timeout。

---

## 3. 超參數嘗試

### 3.1 Scheduler

嘗試導入 cosine decay with warmup，而不是全程使用固定 learning rate。

目前採用：

```yaml
optimizer_lr: 1.0e-4
backbone_lr: 1.0e-4
scheduler_warmup_steps: 5000
scheduler_decay_steps: 100000
scheduler_decay_lr: 1.0e-5
```

Warmup 用來降低長序列訓練初期的不穩定；之後逐步降低 learning rate，使後期更新較平滑。
曾使用較長的 `scheduler_decay_steps: 200000`，目前則配合約 100k 至 120k steps 的訓練，
將 decay window 調整為 100k。

### 3.2 Action horizon

測試過：

```text
action_horizon = 1, 2, 4, 8, 16
```

在 `num_shuffles=3` 的自動 checkpoint evaluation 中，各 run 記錄到的最佳 DSR 為：

| Action horizon | 最佳 DSR | 最佳 MSR | 最佳 SR | 最佳 checkpoint |
|---:|---:|---:|---:|---:|
| 1 | 0.4 | 1.0 | 0.4 | 50k |
| 2 | 0.4 | 1.0 | 0.4 | 10k |
| 4 | 0.3 | 1.0 | 0.3 | 10k |
| **8** | **0.5** | **1.0** | **0.5** | **100k** |
| 16 | 0.3 | 1.0 | 0.3 | 10k |

`action_horizon=1` 每幀重新規劃，閉迴路程度最高，但較容易產生動作抖動。較長的 horizon
可讓抓取軌跡更連續，但 `16` 可能因 open-loop 區間太長而降低修正能力。目前 `8` 在兩者之間
取得較好的平衡，因此被選為預設值。

注意：上表的自動 evaluation 每個 checkpoint 只有 10 episodes，且是從多個 checkpoint 中挑選
最佳值，可能有抽樣誤差與 selection bias。`a8` 另一次 20 episodes evaluation 得到
`DSR=0.4`、`MSR=0.85`。因此目前只能說 `8` 是現有測試中最有希望的設定，還需要更多 seeds
與更多 episodes 才能確認統計顯著性。

### 3.3 Current observation frames

測試過：

```text
current_obs_frames = 0, 2
```

- `0`：action head 只使用 LSTM hidden state。
- `2`：action head 額外取得最近兩個 observation feature。

加入近期觀測的原意，是讓 LSTM 專注保存長期的球與杯子資訊，而 action head 使用近期畫面完成
精細控制。不過這也會增加 action head 輸入維度、顯存需求與學習難度。現有實驗中未觀察到
`2` 穩定超越 `0`，因此目前保留較簡單的 `current_obs_frames=0`。

### 3.4 Sequence length 與 observation stride

測試過兩組主要設定：

| `seq_len` | `obs_stride` | 有效覆蓋範圍 | 特性 |
|---:|---:|---:|---|
| 96 | 8 | 761 frames | 計算量較低，但時間取樣較稀疏 |
| **230** | **3** | **688 frames** | 時間解析度較高，可觀察更多洗牌過程 |

有效覆蓋範圍計算方式：

```text
(seq_len - 1) * obs_stride + 1
```

兩組設定都能覆蓋約 600 幀的完整 episode。`96/8` 較省記憶體，但可能跳過杯子交換期間的重要
中間狀態；`230/3` 雖然 BPTT 較長，卻能以較密集的時間取樣追蹤杯子，因此目前採用後者。

### 3.5 Hidden size

測試過：

```text
hidden_size = 256, 512
```

`256` 的訓練與推論成本較低，但記憶容量也較小。ShellBench 需要同時保留目標杯資訊、洗牌歷史
和 robot state，目前 `512` 的容量較合適，且在可接受的 GPU 記憶體範圍內，因此採用 `512`。

### 3.6 Vision backbone

測試過：

```text
vision_backbone = resnet18, resnet50
```

ResNet50 具有更大的視覺容量，但對 128 x 128 的雙相機長序列而言，顯存與計算成本明顯增加，
不一定能轉化為更好的杯子追蹤結果。ResNet18 訓練較穩定、速度較快，也較容易支援
`seq_len=230`，因此目前使用 ImageNet 預訓練的 ResNet18。

### 3.7 其他資源與穩定性設定

實驗過程中也調整過以下設定：

- `image_size`：早期使用 96，部分設定曾使用 160，目前採用 128。
- `batch_size`：曾使用 2，長序列設定下改為 1，以避免 OOM。
- `use_gradient_checkpointing`：保持啟用，以重算換取顯存。
- `num_workers`：目前使用 4。
- `save_freq`：由 10k 改為 5k，以更密集地觀察 checkpoint 表現。
- `steps`：由 100k 延長到 120k。
- `pretrained_backbone_weights`：使用 `IMAGENET1K_V1`，避免從零學習視覺特徵。
- `image dataset`：將影片預解碼成圖片，避免每個 training step 重複解碼 AV1。

---

## 4. 目前採用的最佳設定

綜合現有結果、訓練穩定性與資源需求，目前所有 `policies/lstm/configs/*.yaml` 採用以下共同設定：

```yaml
policy:
  type: lstm

  vision_backbone: resnet18
  pretrained_backbone_weights: IMAGENET1K_V1
  image_size: 128

  hidden_size: 512
  num_lstm_layers: 2
  state_feature_dim: 128
  dropout: 0.1

  seq_len: 230
  obs_stride: 3
  current_obs_frames: 0
  action_horizon: 8

  use_gradient_checkpointing: true

  optimizer_lr: 1.0e-4
  backbone_lr: 1.0e-4
  optimizer_weight_decay: 1.0e-4
  scheduler_warmup_steps: 5000
  scheduler_decay_steps: 100000
  scheduler_decay_lr: 1.0e-5

batch_size: 1
steps: 120000
seed: 1000
num_workers: 4
save_freq: 5000
log_freq: 100
save_checkpoint: true
```

選擇這組設定的主要原因：

1. `action_horizon=8` 在現有 action-horizon sweep 中得到最高的最佳 DSR。
2. `seq_len=230, obs_stride=3` 保留較密集的洗牌時間資訊，同時覆蓋完整 episode。
3. `hidden_size=512` 提供足夠的長期記憶容量。
4. ResNet18 能兼顧視覺特徵品質、訓練速度和顯存需求。
5. `current_obs_frames=0` 架構較簡單，目前沒有證據顯示加入兩個近期 frame 能穩定提升表現。
6. Cosine decay with warmup 比固定 learning rate 更適合長時間訓練。

---

## 5. 結論與限制

目前最好的實驗 run 是：

```text
outputs/lstm/num_shuffles-3-a8
```

其 100k checkpoint 在 10 episodes 自動評估中得到：

```text
DSR   = 0.5
MSR   = 1.0
SR    = 0.5
kappa = 0.25
```

這表示 action chunk 長度為 8 的 LSTM，在目前實驗中兼顧了選杯與操作能力。因此已將這套設定
同步至其他 LSTM YAML，作為後續 `num_shuffles`、`num_cups` 和 `shuffle_speed` 實驗的共同基準。

不過，現有 evaluation 數量仍偏少，checkpoint 間的 DSR 波動也很明顯。正式宣稱此設定優於其他
組合前，應固定 evaluation protocol，使用多個 seeds 並將每組 evaluation 增加至至少 50 episodes。
因此本文的「最佳」是指**目前已完成實驗中的最佳工作設定**，而不是已經統計證明的全域最佳解。
