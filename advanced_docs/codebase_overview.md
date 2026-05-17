# Codebase 架構總覽

本專案是一個 **Sim-to-Real 模仿學習 (Imitation Learning) 管線**，用於機器人操作任務。完整流程為：用 UMI 錄製人類示範 → SLAM 處理示範資料 → 在 Isaac Lab 模擬器中生成合成資料 → 用 LeRobot 訓練 Diffusion Policy → 在模擬器中評估策略。

平台限定 Linux，主要依賴 NVIDIA Isaac Sim、Isaac Lab、LeRobot。

---

## 目錄結構

```
Advanced-level/
├── packages/                  # 兩個核心子套件
│   ├── umi/                   # UMI 資料處理管線
│   └── simulator/             # Isaac Lab 模擬器任務定義
├── scripts/                   # 入口腳本（資料生成、遙操作、rollout）
├── configs/                   # Shell Game benchmark 的實驗配置（規劃中）
├── umi_pipeline_configs/      # UMI SLAM pipeline 的 YAML 配置
├── advanced_docs/             # 進階文件（任務描述、實作計劃）
├── docs/                      # 使用說明文件
├── tests/                     # 測試
├── data/                      # 資料存放區（.gitkeep）
├── checkpoints/               # 模型 checkpoint 存放區（.gitkeep）
├── private_tasks/             # 自訂私有 task 存放區
├── dependencies/              # Git submodule（IsaacLab）
├── Dockerfile                 # Isaac Sim Docker 容器定義
├── Makefile                   # 建構、啟動 Docker 的指令集
├── pyproject.toml             # UV workspace 根設定
└── uv.lock                    # 統一的鎖定檔
```

---

## 套件管理

專案使用 **uv** 作為套件管理工具，採用 **workspace** 架構：

- **根 `pyproject.toml`**：定義 workspace，`packages/umi` 為 workspace member。根套件本身 `package = false`，僅用於管理依賴。
- **`packages/umi/pyproject.toml`**：UMI 套件，依賴 Click、OpenCV、NumPy、SLAM 相關工具等，安裝後提供 `umi` CLI 指令。
- **`packages/simulator/pyproject.toml`**：Simulator 套件，依賴 `leisaac`（LightwheelAI 的 Isaac Lab 封裝）和 `lerobot`。此套件在 Docker 容器內安裝，不在 host workspace 中。

安裝方式：
- Host 上：`uv sync --package umi`（安裝 UMI 管線）
- Docker 內：`pip install -e packages/simulator`（安裝模擬器套件）

---

## 核心套件詳解

### 1. `packages/umi/` — UMI 資料處理管線

UMI（Universal Manipulation Interface）負責從 GoPro 錄製的人類示範影片中，透過 SLAM 和 ArUco 標記檢測，提取出物件的空間姿態。

#### CLI 入口 (`src/umi/cli.py`)

提供三個指令：

| 指令 | 功能 |
|------|------|
| `umi run-slam-pipeline <config.yaml>` | 執行完整的 SLAM 資料處理管線 |
| `umi visualize-slam-gui <video>` | 啟動 ORB-SLAM3 GUI 除錯工具 |
| `umi merge-object-poses <dir1> <dir2>` | 合併兩個 session 的 `object_poses.json` |

#### Pipeline Executor (`src/umi/pipeline_executor.py`)

管線執行器，讀取 YAML 配置檔，按順序執行各個 service stage。核心特性：

- **配置傳播**：前一個 stage 的配置會自動傳遞給下一個 stage（`inherit_config`）
- **動態載入**：透過 `instance` 欄位動態 import service class
- **Profiler 支援**：可記錄每個 stage 的執行時間

#### Services (`src/umi/services/`)

每個 service 繼承 `BaseService` 抽象類別，實作 `execute()` 方法。管線中的 stage 包括：

| Stage | Service | 功能 |
|-------|---------|------|
| `00_process_video` | `VideoOrganizationService` | 整理原始影片檔案 |
| `01_extract_gopro_imu` | `IMUExtractionService` | 從 GoPro 影片提取 IMU 數據（加速度、陀螺儀等） |
| `02_create_map` | `SLAMMappingService` | ORB-SLAM3 建圖（在 Docker 容器中執行） |
| `03_batch_slam` | `SLAMMappingService` | ORB-SLAM3 批次定位 |
| `04_detect_aruco` | `ArucoDetectionService` | 偵測影片中的 ArUco 標記 |
| `05_run_calibrations` | `CalibrationService` | 相機校正 |
| `05b_verify_calibration` | `CalibrationVerificationService` | 校正結果驗證（僅 verify pipeline） |
| `06_generate_dataset_plan` | `DatasetPlanningService` | 規劃資料集切分方式 |
| `07_frame_to_pose` | `FrameToPoseService` | 將影格中偵測到的 ArUco 轉換為物件姿態 |
| `08_generate_replay_buffer` | `ReplayBufferService` | 產生 Zarr 格式的 replay buffer |

#### 工具模組 (`src/umi/common/`)

提供底層工具函式：

- `cv_util.py` — OpenCV 影像處理工具
- `pose_util.py`, `pose_trajectory_interpolator.py` — 姿態計算與插值
- `orb_slam_util.py` — ORB-SLAM 相關工具
- `mocap_util.py` — 動作捕捉工具
- `timecode_util.py`, `timestamp_accumulator.py` — 時間碼處理
- `interpolation_util.py` — 通用插值工具

#### Pipeline 配置 (`umi_pipeline_configs/`)

每個 GoPro 相機有獨立的配置，以相機代號區分（C2、C6、C9）：

- `verify_pipeline_C*.yaml`：驗證管線（不含 batch_slam 和後續 stage，增加 GUI 顯示和校正驗證）
- `build_dataset_C*.yaml`：完整建資料集管線

---

### 2. `packages/simulator/` — Isaac Lab 模擬器

基於 NVIDIA Isaac Lab 和 LeIsaac 框架，定義機器人操作任務的模擬環境。

#### 資產 (`assets/`)

- **robots/**：機器人模型
  - `franka.py` — Franka Panda 機械臂的 `ArticulationCfg` 配置
  - `franka.usd`, `so101_follower.usd` — 機器人 USD 模型
  - `meshes/` — 碰撞和視覺 mesh（OBJ / STL）
- **scenes/**：場景
  - `kitchen.py`, `dining_room.py`, `living_room.py` — 三個場景的配置
  - 各場景目錄下有 USD 場景檔和物件（杯子、盤子、餐具等）的模型與貼圖

#### 任務定義 (`src/simulator/tasks/`)

遵循一致的架構模式，每個任務包含：

##### Template — 基礎 Franka 任務配置 (`tasks/template/`)

`SingleArmFrankaTaskEnvCfg` 是所有 Franka 單臂任務的基類，定義了：

- **場景**：Franka 機器人 + 腕部攝影機 (wrist) + 前方攝影機 (front) + 燈光
- **觀測空間**：關節位置/速度、上一步動作、兩個攝影機的 RGB 影像
- **動作空間**：7 個關節位置目標 + 1 個夾爪開合指令
- **遙操作設定**：鍵盤/遊戲手把操控，內部使用 Differential IK 轉換成關節目標

##### Cup Stacking (`tasks/cup_stacking/`)

- **任務**：將藍色杯子疊到粉紅杯子上方
- **場景**：廚房場景 + 兩個杯子（BlueCup、PinkCup）
- **成功條件**：藍杯在粉杯正上方（xy 偏差 < 5cm，z 高度差 > 10cm）
- **物件姿態映射**：ArUco tag 1 → blue_cup, tag 2 → pink_cup, anchor tag 0

##### Cutlery Arrangement (`tasks/cutlery_arrangement/`)

- **任務**：將刀叉擺放到正確位置
- **場景**：餐廳場景 + 刀 (knife) + 叉 (fork)
- **物件姿態映射**：ArUco tag 2 → knife, tag 3 → fork

##### Toy Blocks Collection (`tasks/toy_blocks_collection/`)

- **任務**：收集積木
- **場景**：客廳場景 + 三色積木（green_block, blue_block, red_block）
- **物件姿態映射**：ArUco tag 1/2/3 → 三色積木

##### External Task Resolver (`tasks/external.py`)

支援載入外部自訂 task 的機制，接受三種格式：
1. 已註冊的 Gym ID
2. `.py` 檔案路徑（執行後自動 gym.register）
3. `module:Class` 引用

這讓使用者可以在 `private_tasks/` 中定義自己的評估任務。

#### 狀態機 — 自動化資料生成 (`src/simulator/datagen/state_machine/`)

每個任務有對應的 State Machine，用腳本化的方式控制機器人完成任務，自動生成示範資料。以 `CupStackingStateMachine` 為例：

7 個階段：
1. **移到藍杯上方** — 從初始位置線性插值到藍杯正上方
2. **下降接近** — 降到抓取高度
3. **閉合夾爪** — 夾住藍杯
4. **提起** — 升到安全高度
5. **移到粉杯上方** — 水平移動到粉杯位置
6. **下降放置** — 降到堆疊高度後鬆手
7. **後退** — 升起並遠離

技術細節：
- 使用 Jacobian + Damped Least Squares (DLS) 做逆運動學 (IK) 計算
- 在世界座標系中追蹤 waypoint，轉換為關節位置目標
- 每階段有固定步數，用 phase counter 驅動狀態切換

#### 工具模組 (`src/simulator/utils/`)

- **`object_poses_loader.py`** — 將 UMI 輸出的 `object_poses.json`（ArUco 座標系）轉換為 Isaac Lab 的世界座標姿態。核心轉換：anchor-frame 的 tvec/rvec → 世界座標的 (position, quaternion)。刻意不依賴 NumPy/Torch，可獨立單元測試。
- **`domain_randomization.py`** — 光照條件隨機化（強度、色溫、HDR 貼圖），用於 Domain Randomization 增強資料多樣性。

#### 輸入裝置 (`src/simulator/devices/`)

`FrankaKeyboard` — Franka 的鍵盤遙操作裝置。接收鍵盤按鍵的 SE(3) delta，透過 Differential IK 控制器即時解算成 7 軸關節目標 + 夾爪開合指令。

按鍵對應：
- W/S/A/D：前後左右平移
- J/K：上下平移
- H/L/U/I/Q/E：Roll/Pitch/Yaw 旋轉
- C/M：夾爪開/合

---

## 入口腳本 (`scripts/`)

### `scripts/datagen/generate.py` — 合成資料生成

用 State Machine 自動化產生訓練資料的主入口。

**流程：**
1. 根據 `--task` 從 `TASK_REGISTRY` 查找對應的 StateMachine
2. 讀取 `--object_poses` 的 JSON，每個 `status=="full"` 的 entry 產生一個 episode
3. 每個 episode：設定物件姿態 → State Machine 驅動機器人完成任務 → 錄製 observation 與 action
4. 支援 HDF5 或 LeRobot 格式輸出，可上傳至 Hugging Face Hub

**關鍵參數：**
- `--task`：任務 ID（如 `HCIS-CupStacking-SingleArm-v0`）
- `--object_poses`：UMI 產出的物件姿態 JSON
- `--record` / `--use_lerobot_recorder`：啟用錄製功能
- `--step_hz`：模擬頻率（預設 60Hz）

### `scripts/rollout.py` — 策略評估

載入已訓練的 LeRobot policy checkpoint，在 Isaac Lab 中評估機器人表現。

**核心元件：**
- `LeRobotSyncPolicy`：在同一 process 中執行 LeRobot 推論，處理 observation → action 的完整管線
- 支援 Franka Panda 和 SO101 兩種機器人
- 自動設定雙視窗（腕部攝影機 + 前方攝影機）
- 支援 `--task` 接受 external task resolver 的三種格式

**評估指標：**
- 成功率 = 成功 episode 數 / 總 episode 數
- 可設定 episode 長度和評估輪數

### `scripts/teleop.py` — 人工遙操作

讓人類透過鍵盤、遊戲手把或實體 SO101 Leader 臂操控模擬器中的機器人，用於：
- 手動錄製示範資料
- 除錯和測試環境
- 驗證物件姿態是否正確

支援多種輸入裝置：keyboard、gamepad、so101leader、bi-so101leader、lekiwi 系列。

---

## Docker 環境

### Dockerfile

基於 `nvidia/cuda:12.8.1-devel-ubuntu22.04`，安裝：
- Python 3.11
- PyTorch 2.7.0 (CUDA 12.8)
- Isaac Sim 5.1.0
- Isaac Lab（透過 git submodule，路徑 `dependencies/IsaacLab`）
- simulator 套件

### Makefile

| 指令 | 功能 |
|------|------|
| `make install` | 安裝 UMI 套件（host 端） |
| `make install-dev` | 安裝開發依賴（含 pytest、ruff） |
| `make test` | 執行測試 |
| `make build-isaaclab` | 建構 Docker image |
| `make launch-isaaclab` | 建構並啟動 Isaac Lab 容器（預設） |
| `make launch-isaaclab-glowsai-4090` | GlowsAI RTX 4090 環境 |
| `make launch-isaaclab-glowsai-l40s` | GlowsAI L40S 環境（VirtualGL） |
| `make check-isaaclab-gpu` | GPU 環境檢查 |

容器內會掛載整個專案到 `/workspace/aicapstone`，並設定 Vulkan、X11 顯示等。

---

## 測試 (`tests/`)

| 測試檔 | 測試內容 |
|--------|---------|
| `test_repo_layout.py` | 驗證專案結構和必要檔案存在 |
| `test_external_task_resolver.py` | 測試 external task 載入機制的各種 edge case |
| `test_object_poses_loader.py` | 測試 UMI → Isaac Lab 的座標轉換 |
| `test_rollout_wiring.py` | 測試 rollout 腳本的接線邏輯 |
| `tests/fixtures/external_tasks/` | 外部 task 的測試 fixture（有效/無效/多重註冊等） |
| `tests/smoke/` | Smoke test（如 cup_stacking 整合測試） |

UMI 套件也有自己的測試在 `packages/umi/tests/`，涵蓋各 service 和 pipeline executor。

---

## 進階任務 — ShellBench (`advanced_docs/`)

專案正在規劃一個新的 benchmark 任務：**ShellBench（猜杯子遊戲）**，用來測量 visuomotor policy 的長期記憶能力。

### 概念

Shell game 是一個 **non-Markovian** 任務 — 球被蓋住後，三個杯子外觀相同，policy 必須記住之前看到球在哪，才能選對杯子。

### 四階段流程

| Phase | 名稱 | 內容 | 控制方式 |
|-------|------|------|---------|
| 1 | Reveal | 球出現在某杯旁，camera 觀察 | 環境 scripted |
| 2 | Cover | 杯子蓋住球 | 環境 scripted |
| 3 | Shuffle | 杯子兩兩交換 N 次 | 環境 scripted |
| 4 | Act | 機器人掀開一個杯子 | Policy 控制 |

### 可控難度參數

- 杯子數量（3/4/5）
- 交換次數（0~5+）
- 交換速度

### 評估指標

- **DSR**（Decision Success Rate）：是否選對杯子
- **MSR**（Manipulation Success Rate）：是否成功掀起杯子
- **SR**（Success Rate）：選對且掀起
- **κ**（Cohen's Kappa）：校正隨機猜對的因素

詳見 `advanced_docs/task_description.md` 和 `advanced_docs/IMPLEMENT.md`。

---

## 完整資料流程

```
[真實世界]                    [模擬器 (Docker)]               [Host]
                                                              
GoPro 錄影                                                    
    │                                                         
    ▼                                                         
UMI Pipeline ──────────────────────────────────────────┐      
  00. 影片整理                                          │      
  01. IMU 提取                                          │      
  02. SLAM 建圖                                         │      
  03. SLAM 定位                                         │      
  04. ArUco 偵測                                        │      
  05. 相機校正                                          │      
  06. Dataset 規劃                                      │      
  07. Frame → Pose                                      │      
  08. Replay Buffer                                     │      
    │                                                   │      
    ▼                                                   │      
object_poses.json ─────────► generate.py ──────────────►│      
                              (State Machine            │      
                               自動操作機器人)           │      
                                  │                     │      
                                  ▼                     │      
                            LeRobot Dataset ───────────►│      
                                                        │      
                                                  lerobot-train
                                                        │      
                                                        ▼      
                                                   checkpoint  
                                                        │      
                            rollout.py ◄────────────────┘      
                              (載入 policy                      
                               模擬器評估)                      
```
