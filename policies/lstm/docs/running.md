# 怎麼訓練與評估 LSTM Policy（執行手冊）

這份是「實際動手跑」的指令手冊。背景與架構看
[`how_it_works.md`](./how_it_works.md) 與 [`implementation_plan.md`](./implementation_plan.md)。

> **所有指令都在 Docker 容器內跑**（lerobot + Isaac Sim 都在那）。
> 進容器：host 端 `make launch-isaaclab-glowsai-4090`（或你用的 launch target）。
> repo 掛在 `/workspace/aicapstone`。

---

## 0. 先確認 plugin 沒壞（幾秒）

```bash
cd /workspace/aicapstone
python policies/lstm/scripts/smoke_test.py        # 看到 ✅ ALL SMOKE TESTS PASSED 即可
```

---

## 0.5 資料準備的兩個已知雷（先看，省得重踩）

1. **HF dataset 沒打版本 tag** → lerobot 線上抓會報 `RevisionNotFoundError`。
   繞法：先手動把 dataset 下載到 lerobot 本地 cache，訓練就會讀本地、跳過 tag 檢查：
   ```bash
   # 容器內，需登入 HF（export HF_TOKEN=... 或 hf auth login）避免限流
   python - <<'PY'
   from huggingface_hub import snapshot_download
   repo = "johnnyli1220/shellbench-num_shuffles-0"
   snapshot_download(repo, repo_type="dataset",
                     local_dir=f"/root/.cache/huggingface/lerobot/{repo}")
   PY
   ```
   一勞永逸版：請 dataset 擁有者（johnnyli1220）幫每個 repo 打 `v3.0` tag，之後就能直接線上抓。
```
python -c "
from huggingface_hub import snapshot_download
snapshot_download('johnnyli1220/shellbench-num_shuffles-0',
                  repo_type='dataset',
                  local_dir='/home/youzhe0305/.cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-0')
" 
```


2. **影片解碼後端** → 一定要加 `--dataset.video_backend=pyav`（容器缺 FFmpeg，torchcodec 載不起來）。

```
python policies/lstm/scripts/decode_dataset_to_images.py \
  --src-repo johnnyli1220/shellbench-num_shuffles-0 \
  --src-root /home/youzhe0305/.cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-0 \
  --dst-root data/lerobot_img/johnnyli1220/shellbench-num_shuffles-0
```

## 1. 訓練

### 建議配置（含 cup classification head）

```bash
cd /workspace/aicapstone
python policies/lstm/scripts/train_lstm.py \
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
  --output_dir=outputs/lstm/ns0_160px_cup_v1 \
  --job_name=lstm_ns0_160px_cup
```

> **用 decode 過的 image 版 dataset**（`--dataset.root` 指向 `data/lerobot_img/...`），
> 不用加 `--dataset.video_backend=pyav`。如果用原始 video dataset，才需要加 pyav backend。

> **長時間訓練用 tmux**：容器內 `apt-get install -y tmux`，
> `tmux new -s train` 開 session，`Ctrl+B D` 離開，`tmux attach -t train` 回去。

訓練完，checkpoint 會在 `outputs/lstm/ns0_v1/checkpoints/` 底下。
確認實際路徑（lerobot 會建一個 `last` 指向最後一個 step）：

```bash
ls -R outputs/lstm/ns0_v1/checkpoints/ | head
# 預期看到 .../checkpoints/last/pretrained_model/（含 model.safetensors + config.json + 前後處理器）
```

### 可調旋鈕（記憶體 / 記憶長度）

| flag | 預設 | 說明 |
|------|------|------|
| `--policy.seq_len` | 96 | 每個訓練樣本取幾個時間點（= BPTT 長度）。有效覆蓋長度是 `(seq_len-1)*obs_stride+1` 幀。 |
| `--policy.obs_stride` | 8 | 每隔幾幀取一次觀測。`seq_len=96, obs_stride=8` 覆蓋 761 幀；eval policy 會用同一個 stride 更新 hidden state。 |
| `--policy.image_size` | 96 | 影像 resize 邊長。降低最省記憶體。 |
| `--policy.hidden_size` | 512 | LSTM 隱藏維度。 |
| `--policy.use_gradient_checkpointing` | true | 用重算換記憶體，長序列必開。 |
| `--batch_size` | (lerobot 8) | **OOM 就先降這個**（2 → 1）。 |
| `--steps` | (lerobot 100k) | 第一次先設小（如 5000）確認跑得動，再加大。 |

### 💥 OOM（爆顯存）怎麼辦（依序試）

1. `--batch_size=1`
2. `--policy.image_size=64`
3. 提高 `--policy.obs_stride` 或調小 `--policy.seq_len`；注意有效覆蓋長度要蓋過 reveal→act
4. 確認 `--policy.use_gradient_checkpointing=true`

### ⚡ 加速：把 dataset 解碼成 image（一次性、持久化、可重用）

dataset 的相機是 AV1 影片，訓練每步都要 CPU 解碼 → 嚴重 CPU-bound（GPU 在等，~1.2s/步）。
要練到 grasp 需要的 ~10 萬步，**先把影片解碼成圖片**（之後訓練不用解碼 → GPU-bound、快很多）：

```bash
python policies/lstm/scripts/decode_dataset_to_images.py \
  --src-repo johnnyli1220/shellbench-num_shuffles-3 \
  --src-root /workspace/aicapstone/data/raw/johnnyli1220/shellbench-num_shuffles-3 \
  --dst-root /workspace/aicapstone/data/lerobot_img/johnnyli1220/shellbench-num_shuffles-3 \
  --resize 160
```

> **`--resize 160` 建議加上**：存 160px 圖片（省空間也省 dataloader RAM）。
> 模型內部的 `F.interpolate` 會再 resize 到 `--policy.image_size`，所以 decode 的解析度
> 只要 >= 訓練用的 image_size 就好。

下載 dataset 到掛載點（容器外或容器內皆可）：
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('johnnyli1220/shellbench-num_shuffles-3',
                  repo_type='dataset',
                  local_dir='/workspace/aicapstone/data/raw/johnnyli1220/shellbench-num_shuffles-3')
"
```

- **`--dst-root` 一定放 `data/` 底下**（= host 掛載點、持久化、已 .gitignore）。
  **不要放 `/root/.cache`** —— 容器 `--rm`，cache 重開就沒了，白轉。
- 一次性 ~30-40 分、每個 variant ~7GB；轉好全組重用。
- 訓練時加 `--dataset.root=<那個 data/ 路徑>` 即用 image 版。
- 確認生效：訓練的 `data_s` 應從 ~1.2 掉到接近 0（GPU-bound）。

---

## 2. 評估（算 DSR / MSR / SR / κ）

用 LSTM 評估 wrapper（零侵入跑既有 `eval_shell_game.py`）：

```bash
cd /workspace/aicapstone
python policies/lstm/scripts/eval_lstm.py \
  --task HCIS-ShellGame-SingleArm-v0 \
  --device cuda --enable_cameras --headless \
  --policy_backend lerobot \
  --policy_type lerobot-lstm \
  --policy_checkpoint_path outputs/lstm/ns0_160px_cup_v1/checkpoints/last/pretrained_model \
  --policy_action_horizon 1 \
  --num_episodes 20 \
  --num_cups 3 --num_shuffles 0 \
  --reveal_frames 50 --cover_frames 10 --shuffle_per_swap_frames 30 --act_frames 150 \
  --max_act_steps 600 \
  --output_json outputs/lstm/ns0_160px_cup_v1/metrics.json
```

> `--max_act_steps`（預設 600）：沒夾起杯子的 episode 會在 act 階段跑滿這麼多步就判 MISS 結束，
> 不再空跑到 episode time_out（~3600 步），eval 快很多。成功抓取約 360 步內完成、不受影響；
> 想更快可調小（如 400），但太小會誤砍掉慢一點的成功抓取（MSR 假性下降）。

> **用 `--headless`（推薦）：** 實測 headless 能跑且輸出乾淨（會到 `Starting evaluation` →
> `Episode N/20`）。不加 `--headless` 也能跑，但會噴一堆 `vkCreateSwapchainKHR failed` /
> `backbuffers are not initialized` 警告（想開視窗失敗的雜訊，不影響相機離線渲染與結果）。
> 註：VNC 看不到 Isaac 視窗是正常的——eval 用的是離線相機渲染，不需要可見視窗。
> 另外：**評估前先停掉訓練**，兩個都吃 GPU，一起跑會搶資源。

結果在 `outputs/lstm/ns0_v1/metrics.json`（含 DSR/MSR/SR/κ 與每集細節）。

> `--policy_action_horizon 1`：LSTM 是逐步推論，每幀輸出一個動作再重新觀測。

### ⚠️ 評估的階段幀數要對得上 dataset 怎麼生的

`--reveal_frames / --cover_frames / --shuffle_per_swap_frames / --act_frames` 必須跟組員
**生這份 dataset 時用的設定一致**，否則任務節奏會不同。上面用的是 `base.yaml` 預設值；
若不確定，跟 johnnyli1220 確認當初 `generate_shell_game.py` 的階段參數（量到的平均 episode
長度是 ~514 幀，代表 act/FSM 階段不短，`act_frames` 要夠大讓掀杯動作完成）。

---

## 3. 判讀第一個結果（打通的目標）

| 指標 | 期待（num_shuffles=0） | 意義 |
|------|----------------------|------|
| **MSR** | 高（接近 1） | 有沒有成功掀起某個杯子（操作能力） |
| **DSR** | **明顯 > 1/3（chance）** | 有沒有選對杯子（記憶能力） |
| **κ** | > 0 | 校正亂猜後的記憶能力 |

ns=0 球幾乎不動，記憶很簡單，所以 **DSR 應該明顯高於 1/3**。
- 若 DSR ≈ 1/3：plugin 雖然跑得動，但沒學到東西 → 查 loss 有沒有下降、影像有沒有正確進模型。
- 若 DSR 明顯 > 1/3：**整合與學習都成立，可以進 Phase 3** —— 換 num_shuffles=1..5 訓練評估，畫記憶衰減曲線。

---

## 4. 之後：掃整條曲線（Phase 3，尚未做）

> ⚠️ **先跟組員對齊訓練 protocol**（見 `implementation_plan.md` §7.5）。
> 主結果用 **Protocol A**：每個難度各訓一個 policy、在同難度評估（要跟 ACT/DP/VQ-BeT 同 protocol 才能比）。
> ns=0 這個 de-risk run 學到的有限是正常的 —— 真正跑時每個難度的 dataset 都含該難度的洗牌，policy 會學到對應記憶。

把上面 1+2 對 `num_shuffles ∈ {0,1,2,3,4,5}` 各跑一次（換 `--dataset.repo_id` 與
`--num_shuffles`），收集每個的 DSR，畫 DSR vs. num_shuffles，跟 ACT/Diffusion/Oracle 疊在一起。
（每個 variant 同樣要先準備 dataset，見 §0.5。）

> 自動化：可寫一份 `policies/lstm/configs/` 的 sweep，或擴充既有 `scripts/run_sweep.py`。
> 注意 `run_sweep.py` 的 eval 是寫死呼叫 `scripts/eval_shell_game.py`（沒套我們的 patch），
> 所以全自動 sweep 需要 (a) 讓 eval import register，或 (b) sweep 改呼叫 `eval_lstm.py`。
> 這個整合留到 Phase 3 決定。
