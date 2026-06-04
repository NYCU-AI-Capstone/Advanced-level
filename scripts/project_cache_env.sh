# Source this file before running tools that download data directly, such as
# `lerobot-train`, `hf download`, or `wandb`.
#
# Usage:
#   cd /project/youzhe0305/ai-capstone/Advanced-level
#   source scripts/project_cache_env.sh

_AICAPSTONE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AICAPSTONE_CACHE_DIR="${AICAPSTONE_CACHE_DIR:-${_AICAPSTONE_PROJECT_ROOT}/.cache}"

export HF_HOME="${AICAPSTONE_CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${AICAPSTONE_CACHE_DIR}/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${AICAPSTONE_CACHE_DIR}/huggingface/hub"
export HF_DATASETS_CACHE="${AICAPSTONE_CACHE_DIR}/huggingface/datasets"
export HF_LEROBOT_HOME="${AICAPSTONE_CACHE_DIR}/huggingface/lerobot"
export HF_LEROBOT_CALIBRATION="${AICAPSTONE_CACHE_DIR}/huggingface/lerobot/calibration"
export TRANSFORMERS_CACHE="${AICAPSTONE_CACHE_DIR}/huggingface/transformers"
export TORCH_HOME="${AICAPSTONE_CACHE_DIR}/torch"
export XDG_CACHE_HOME="${AICAPSTONE_CACHE_DIR}/xdg"
export WANDB_DIR="${AICAPSTONE_CACHE_DIR}/wandb/runs"
export WANDB_CACHE_DIR="${AICAPSTONE_CACHE_DIR}/wandb/cache"
export WANDB_CONFIG_DIR="${AICAPSTONE_CACHE_DIR}/wandb/config"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_LEROBOT_HOME}" \
  "${HF_LEROBOT_CALIBRATION}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}"

unset _AICAPSTONE_PROJECT_ROOT
