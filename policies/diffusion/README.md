# Diffusion Policy Evaluation

Evaluate a trained Diffusion Policy checkpoint in headless Isaac Sim with one command:

```bash
python policies/diffusion/scripts/eval_diffusion.py \
  --run_dir outputs/diffusion/shellbench-num_shuffles-3 \
  --num_episodes 100 \
  --seed 529
```

The wrapper automatically:

- switches to the repository `.venv` created by `uv`;
- selects `checkpoints/best`, then `checkpoints/last`, then the latest numeric checkpoint;
- reads `train_config.json` and `config.json` from the selected checkpoint;
- infers `num_shuffles`, `num_cups`, and `shuffle_speed` from the run and dataset names;
- enables cameras and runs Isaac Sim with `--headless`;
- saves results to `<run_dir>/metrics.json`.

Evaluate a specific checkpoint step:

```bash
python policies/diffusion/scripts/eval_diffusion.py \
  --run_dir outputs/diffusion/shellbench-num_shuffles-0 \
  --checkpoint_step 60000 \
  --num_episodes 100
```

Inspect the generated evaluation arguments without launching Isaac Sim:

```bash
python policies/diffusion/scripts/eval_diffusion.py \
  --run_dir outputs/diffusion/shellbench-num_shuffles-0 \
  --num_episodes 100 \
  --dry_run
```

The LSTM wrapper provides the same compact interface:

```bash
python policies/lstm/scripts/eval_lstm.py \
  --run_dir outputs/lstm/shuffle_speed-2.0 \
  --num_episodes 100
```
