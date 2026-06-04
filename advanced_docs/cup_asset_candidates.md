# 杯子 Asset 替換筆記

記錄日期：2026-05-30

## 目標

把 cup_stacking 任務的兩個杯子換成外部 USD asset。挑選條件是無把手、不要馬克杯，
高度和形狀盡量貼近原本的杯子，材質偏好不透明，並且保留藍／粉兩色方便區分。

## 原本杯子的規格

實際被任務用到的是 kitchen 場景這兩個檔：

- `packages/simulator/assets/scenes/kitchen/objects/BlueCup/BlueCup.usd`
- `packages/simulator/assets/scenes/kitchen/objects/PinkCup/PinkCup.usd`

（table 場景底下也各有一份同名檔，但 cup_stacking 沒用到。）

程式從 `packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py:42,50`
兩行的 `usd_path` 載入。

從 USD 量出來的尺寸：BlueCup 直徑約 7.4 cm、高約 9.6 cm；PinkCup 直徑約 8.0 cm、
高約 10.4 cm。兩個都是上寬下窄的錐形、無把手，材質是 OmniPBR 不透明（藍、粉），
PinkCup 還帶一個 textures 資料夾。這兩個檔是在 Omniverse Composer 裡手工做的，
metadata 裡沒有指向任何公開資料集，所以找不到「同一個原始來源」可以直接拿同款。

## 評估過的候選

### P_Glassware_Short（最後採用）

NVIDIA Omniverse 官方 asset server 上的玻璃杯，尺寸最接近原杯。

- URL：`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/ArchVis/Residential/Kitchen/Kitchenware/Dinnerware/P_Glassware_Short.usd`
- 縮圖：同目錄 `.thumbs/256x256/P_Glassware_Short.usd.png`
- 直徑約 8.1 cm、高 9.5 cm，錐形上寬下窄，無把手
- 材質是 OmniGlass.mdl（半透明玻璃，無外部貼圖，檔案自包含）

唯一不符的是它本來是透明玻璃，不是原杯那種不透明塑膠。這點可以靠換材質解決，
見下方「實際做法」。

### P_Glassware_Tall

同一個目錄下的高玻璃杯，一樣無把手，但直徑約 8.3 cm、高 14 cm，偏高也偏直筒，
形狀和高度都離原杯比較遠，所以沒選。

- URL：`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/ArchVis/Residential/Kitchen/Kitchenware/Dinnerware/P_Glassware_Tall.usd`
- 縮圖：同目錄 `.thumbs/256x256/P_Glassware_Tall.usd.png`

### YCB 025_mug（排除）

不透明而且是 SimReady，本來很合適，但有把手，不符條件。

- URL：`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/025_mug.usd`

## 其他可以找不透明無把手杯子的來源

Omniverse 官方 server 的 drinkware 很少，不透明又無把手的杯子沒有現成的，
要找得往外部站。記錄幾個備用：

- YCB benchmark（https://www.ycbbenchmarks.com/）的 object set 065 "Cups" 是一組
  不同顏色、無把手的錐形疊杯，語意上最貼近疊杯任務。格式是 mesh（obj/ply 加貼圖），
  要用 Blender 或 usd tools 轉成 USD。
- Sketchfab（https://sketchfab.com/）搜 paper cup、plastic cup、tumbler，篩
  Downloadable，不透明杯子選擇多。下載 glTF 或 USDZ 後再轉 USD。
- Objaverse（https://objaverse.allenai.org/）物件量大，搜 cup，glTF 格式要轉 USD。

這些外部站幾乎都不是原生 USD，會多一道轉檔。

## 實際做法

最後沒有再去外部站找，而是直接拿 P_Glassware_Short 的幾何，把材質從 OmniGlass
換成不透明 OmniPBR 並染色，省掉找檔和轉檔。結果就是不透明、無把手、尺寸對得上的杯子。

變更內容：

- 原檔先備份成 `BlueCup/BlueCup.orig.usd.bak`、`PinkCup/PinkCup.orig.usd.bak`
- 玻璃幾何放進 `BlueCup/Glassware_Short_geom.usd`、`PinkCup/Glassware_Short_geom.usd`
- 重寫 `BlueCup/BlueCup.usd`、`PinkCup/PinkCup.usd`，用 reference 包住玻璃幾何，
  再疊一層不透明 OmniPBR 材質（藍 (0.05, 0.16, 0.7)、粉 (0.92, 0.42, 0.58)）和 scale
- 檔名和路徑都沒變，所以 cup_stacking_env_cfg.py 不用改

`usdcat --flatten` 兩個檔都能乾淨組合，mesh 也確實綁到新材質。

## 還沒驗證、可能要微調的地方

這台機器沒有 GPU、跑不了 Isaac，下面幾點是從 USD 幾何推算的，要進 sim 才能確認：

1. 尺寸。scale 設成藍 0.0101、粉 0.0109。如果進 sim 後大小明顯不對（最可能是差約
   100 倍的單位問題），改 USD 裡的 `xformOp:scale`，或在 env_cfg 的
   `UsdFileCfg(scale=...)` 補一個係數就好。
2. 原點高度。玻璃杯的原點在底部（z=0），原杯的原點在杯緣，兩者生成高度會不一樣。
   如果杯子浮空或穿桌，調 cup_stacking_env_cfg.py 的 `OBJECT_Z`（目前 0.12）。
3. 疊杯判定。`blue_cup_on_top_pink_cup` 的 height_threshold 可能要跟著原點變化調整。
4. mass。env_cfg 的 mass_props 仍維持 0.1 kg，不受這次替換影響。

## 確認沒問題後可以清掉的東西

- `PinkCup/textures/`：舊粉杯的貼圖，現在用不到
- 兩個 `*.orig.usd.bak`：新杯確認 OK 後可刪
