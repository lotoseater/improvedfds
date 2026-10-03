# Improved FDS Sampling Experiments

---

## Overview

This repository contains an experimental JiT implementation of Flow Divergence
Sampler (FDS), together with additional sampling methods and tools for paired
ImageNet-256 comparisons. The original FDS method is included as the
`heun_ours` baseline; the other experimental methods are documented separately
and should not be confused with the original paper method.

The current implementation uses JiT (Just-image Transformer), a
class-conditional pixel-space image generation model trained on ImageNet-256.

---

## Installation

**Requirements:** Python 3.9 and a CUDA-capable GPU.

```bash
# 1. Clone the repository
git clone git@github.com:lotoseater/improvedfds.git
cd improvedfds

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Sampling

Download a compatible JiT checkpoint separately, then run a small generation
job. Replace the checkpoint and output paths as needed.

```bash
FDS_LOCAL_METRIC=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/main_jit.py \
  --model JiT-B/16 \
  --img_size 256 \
  --resume /path/to/checkpoint-directory \
  --output_dir /path/to/output-directory \
  --evaluate_gen \
  --sampling_method heun_mc_directional \
  --num_sampling_steps 50 \
  --cfg 3.0 \
  --interval_min 0.1 \
  --interval_max 1.0 \
  --num_images 8 \
  --gen_bsz 4 \
  --class_idx 0 \
  --seed 0 \
  --seed_delta 42 \
  --seed_eps 1234 \
  --num_delta 1 \
  --skip_online_metrics
```

## Local Metric MC Comparisons

The existing MC candidate trajectory can be scored with the following local
metrics without changing the checkpoint, Heun update, candidate perturbation,
or random seeds:

```text
official baseline     heun_ours (the existing official FDS implementation)
heun_mc_directional  u_v^T J_v u_v
heun_mc_acceleration ||v_next - v|| / (|dt| ||v|| + eps)
heun_mc_spectral     warm-start power estimate of lambda_max((J_v + J_v^T) / 2)
heun_mc_lipschitz    warm-start power estimate of ||J_v||_2
```

The FDS baseline is not duplicated as another MC method: use the existing
`heun_ours` implementation directly. Run the local metric methods with the
same checkpoint, labels, noise, Heun trajectory, candidate perturbation and
seeds as that baseline. Local metric runs should set `FDS_LOCAL_METRIC=1`;
this keeps the model eager and uses the math attention backend so JVP/VJP are
available:

```bash
FDS_LOCAL_METRIC=1 python src/main_jit.py \
  --sampling_method heun_mc_directional \
  --num_images 10000 \
  --evaluate_gen \
  --resume /path/to/checkpoint_dir \
  --output_dir /path/to/output_dir \
  ...
```

Each run writes raw per-sample/per-timestep records:

```text
<method>-steps50-JiT-B-16-cfg3.0/
└── local_metric_trace_rank000.pt
```

The trace contains `sample_ids`, `t`, `base_metric`,
`candidate_metric`, `selected_metric`, `accepted`, `dt`, and
`power_iterations`. Metric tensors use `[time, batch]` layout and are saved
before any time-layer normalization, so the existing normalization, ranking,
and correlation analysis can consume the raw values. Each trace also includes
pointwise statistics (mean, median, standard deviation, p5/p25/p50/p75/p95,
and max) plus per-trajectory mean, max, and `sum_t metric_t * |dt|`
aggregations.

For `heun_mc_spectral`, the first evaluated metric time point uses 5 power
iterations and every later time point uses 1 warm-start iteration. Each
iteration uses one JVP and one VJP. For `heun_mc_lipschitz`, the same
`5 + 1 + ...` schedule estimates the true local Jacobian spectral norm:

```text
a = J_v @ u
b = J_v^T @ a
u = b / ||b||
L_t = ||a||
```

The Lipschitz path is batch-wise, keeps one right singular-vector estimate per
sample, and never materializes the Jacobian. It does not use Lanczos or
per-sample autograd calls.



---

## Pretrained Checkpoints

Download the pretrained JiT checkpoints and place them in the `checkpoints/` directory as follows:

```
flow-divergence-sampler/
└── checkpoints/
    ├── jit-b-16/
    │   └── checkpoint-last.pth      # JiT-B/16  (~131M params)
    ├── jit-l-16/
    │   └── checkpoint-last.pth      # JiT-L/16  (~131M params)
    └── jit-h-16/
        └── checkpoint-last.pth      # JiT-H/16  (best quality)
```




---

## Acknowledgements

This implementation builds on the original FDS and JiT repositories, as well
as the following projects:

- [Flow Divergence Sampler](https://github.com/yeonwoo378/flow-divergence-sampler)
- [JiT](https://github.com/LTH14/JiT)
- [EDM](https://github.com/NVlabs/edm)
- [FFJORD](https://github.com/rtqichen/ffjord)

---

## Citation

The original FDS method is described in the following paper:

```bibtex
@misc{cha2026trainingfreerefinementflowmatching,
      title={Training-Free Refinement of Flow Matching with Divergence-based Sampling},
      author={Yeonwoo Cha and Jaehoon Yoo and Semin Kim and Yunseo Park and Jinhyeon Kwon and Seunghoon Hong},
      year={2026},
      eprint={2604.04646},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.04646},
}
```

---

## Contact

For any inquiries, please contact **Yeonwoo Cha** at `ckdusdn03@kaist.ac.kr`.
