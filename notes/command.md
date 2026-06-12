```bash
uv sync
source .venv/bin/activate
```

```bash
export OMNI_KIT_ACCEPT_EULA=YES
echo $OMNI_KIT_ACCEPT_EULA
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -c \
  'from isaacsim.kit.kit_app import check_eula; check_eula(); print("EULA OK")'
```

```bash
hf auth login
```

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('johnnyli1220/shellbench-num_cups-5',
                  repo_type='dataset',
                  local_dir='.cache/huggingface/lerobot/johnnyli1220/shellbench-num_cups-5')
" 
```

```bash
python policies/lstm/scripts/decode_dataset_to_images.py \
  --src-repo johnnyli1220/shellbench-num_shuffles-3 \
  --src-root .cache/huggingface/lerobot/johnnyli1220/shellbench-num_shuffles-3 \
  --dst-root data/lerobot_img/johnnyli1220/shellbench-num_shuffles-3 \
  --resize 128
```

```bash
python policies/lstm/scripts/train_lstm.py --config policies/lstm/configs/num_shuffles_3.yaml
```
