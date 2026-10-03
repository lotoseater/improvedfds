#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zjiaak/SSD/projects/fds"
ASSET_ROOT="${FDS_ASSET_ROOT:-/home/zjiaak/HDD/fds}"
OUTPUT_ROOT="${FDS_OUTPUT_ROOT:-/home/zjiaak/HDD/fds/runs}"

MODEL="${MODEL:-JiT-H/16}"
SAMPLING_METHOD="${SAMPLING_METHOD:-heun_ours}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-1.5}"
ITERATIONS="${ITERATIONS:-1}"
PERTURB_SCALE="${PERTURB_SCALE:-1e-2}"
PERTURB_SCHEDULE="${PERTURB_SCHEDULE:-linear}"
ITER_SCHEDULE="${ITER_SCHEDULE:-linear}"
STOP_T="${STOP_T:-0.5}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-64}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-0}"
SEED_DELTA="${SEED_DELTA:-42}"
SEED_EPS="${SEED_EPS:-1234}"
NUM_DELTA="${NUM_DELTA:-1}"

case "${MODEL}" in
    "JiT-B/16")
        CKPT_DIR="${ASSET_ROOT}/checkpoints/jit-b-16"
        ;;
    "JiT-L/16")
        CKPT_DIR="${ASSET_ROOT}/checkpoints/jit-l-16"
        ;;
    "JiT-H/16")
        CKPT_DIR="${ASSET_ROOT}/checkpoints/jit-h-16"
        ;;
    *)
        echo "Unsupported MODEL=${MODEL}" >&2
        echo "Use one of: JiT-B/16, JiT-L/16, JiT-H/16" >&2
        exit 1
        ;;
esac

CHECKPOINT_PATH="${CKPT_DIR}/checkpoint-last.pth"
FID_STATS_PATH="${ASSET_ROOT}/fid_stats/jit_in256_stats.npz"
PRC_FEATURES_PATH="${ASSET_ROOT}/fid_stats/imagenet_val-inception-v3-compat-features-2048.pt"
VIRTUAL_NPZ_PATH="${ASSET_ROOT}/VIRTUAL_imagenet256_labeled.npz"

for required_path in \
    "${CHECKPOINT_PATH}" \
    "${FID_STATS_PATH}" \
    "${PRC_FEATURES_PATH}" \
    "${VIRTUAL_NPZ_PATH}"
do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required asset: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}"

RUN_NAME="${SAMPLING_METHOD}_$(echo "${MODEL}" | tr '/' '-')_steps${NUM_STEPS}_cfg${CFG_SCALE}_iter${ITERATIONS}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" src/main_jit.py \
    --model "${MODEL}" \
    --img_size 256 \
    --resume "${CKPT_DIR}" \
    --output_dir "${RUN_DIR}" \
    --evaluate_gen \
    --sampling_method "${SAMPLING_METHOD}" \
    --num_sampling_steps "${NUM_STEPS}" \
    --cfg "${CFG_SCALE}" \
    --iter "${ITERATIONS}" \
    --perturb_scale "${PERTURB_SCALE}" \
    --perturb_schedule "${PERTURB_SCHEDULE}" \
    --iter_schedule "${ITER_SCHEDULE}" \
    --stop_t "${STOP_T}" \
    --num_images "${NUM_IMAGES}" \
    --gen_bsz "${GEN_BSZ}" \
    --seed "${SEED}" \
    --seed_delta "${SEED_DELTA}" \
    --seed_eps "${SEED_EPS}" \
    --num_delta "${NUM_DELTA}" \
    --skip_online_metrics
