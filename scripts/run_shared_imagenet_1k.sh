#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zjiaak/SSD/projects/fds"
TEMPLATE_DIR="${FDS_TEMPLATE_DIR:-/home/zjiaak/HDD/fds/templates/imagenet256_shared_1k_v2}"
OUTPUT_DIR="${FDS_OUTPUT_DIR:-/home/zjiaak/HDD/fds/runs/shared_imagenet_1k_v2}"
CHECKPOINT_DIR="${FDS_CHECKPOINT_DIR:-/home/zjiaak/HDD/fds/checkpoints/jit-b-16-official}"
BSZ="${FDS_GEN_BSZ:-4}"
NUM_IMAGES="${FDS_NUM_IMAGES:-1000}"
METHODS="${FDS_METHODS:-heun heun_ours heun_lip_min_3candidates heun_fds_gd heun_randdir}"

cd "${REPO_ROOT}"

python scripts/run_shared_imagenet_compare.py \
    --prepare \
    --template-dir "${TEMPLATE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-images "${NUM_IMAGES}"

gpu=0
pids=()
for method in ${METHODS}; do
    log_file="${OUTPUT_DIR}/${method}.log"
    mkdir -p "${OUTPUT_DIR}"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" FDS_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 \
        python scripts/run_shared_imagenet_compare.py \
            --method "${method}" \
            --template-dir "${TEMPLATE_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --gpu 0 \
            --batch-size "${BSZ}" \
            --num-images "${NUM_IMAGES}" \
            --overwrite \
            > "${log_file}" 2>&1
    ) &
    pids+=("$!")
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
