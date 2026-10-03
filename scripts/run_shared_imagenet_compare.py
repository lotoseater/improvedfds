#!/usr/bin/env python3
"""Run ImageNet samples from a fixed, reusable input tensor.

The template stores the initial latent noise, labels, and a fixed random
direction independently from generated images. Every method reads the same
stored tensors, so later sampler comparisons do not depend on RNG ordering.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

# FDS-GD needs higher-order autograd; the current torch.compile donated-buffer
# path is incompatible with that backward pass. Keep all comparison methods
# on the same eager implementation.
os.environ.setdefault("FDS_DISABLE_TORCH_COMPILE", "1")

from denoiser import DenoiserCustom


DEFAULT_TEMPLATE_DIR = Path("/home/zjiaak/HDD/fds/templates/imagenet256_shared_1k_v2")
DEFAULT_RUN_DIR = Path("/home/zjiaak/HDD/fds/runs/shared_imagenet_1k_v2")
DEFAULT_CHECKPOINT_DIR = Path("/home/zjiaak/HDD/fds/checkpoints/jit-b-16-official")
DEFAULT_METHODS = (
    "heun",
    "heun_ours",
    "heun_lip_min_3candidates",
    "heun_fds_gd",
    "heun_randdir",
)

METHOD_DEFAULT_NUM_DELTA = {
    "heun_lip_min_3candidates": 3,
}


def build_args(method: str, output_dir: str, batch_size: int, args):
    return SimpleNamespace(
        model="JiT-B/16",
        img_size=256,
        attn_dropout=0.0,
        proj_dropout=0.0,
        class_num=1000,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        noise_scale=1.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method=method,
        num_sampling_steps=50,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        iter=args.iter,
        perturb_scale=args.perturb_scale,
        perturb_schedule=args.perturb_schedule,
        iter_schedule="linear",
        stop_t=args.stop_t,
        seed_delta=args.seed_delta,
        seed_eps=args.seed_eps,
        num_delta=METHOD_DEFAULT_NUM_DELTA.get(method, args.num_delta),
        marginal_t_min=1e-3,
        marginal_b_min=1e-6,
        marginal_quad_weight=1.0,
        marginal_div_weight=1.0,
        output_dir=output_dir,
        gen_bsz=batch_size,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_template(args):
    args.template_dir.mkdir(parents=True, exist_ok=True)
    noise_path = args.template_dir / "initial_noise.pt"
    labels_path = args.template_dir / "labels.pt"
    direction_path = args.template_dir / "fixed_direction.pt"
    bundle_path = args.template_dir / "shared_inputs.pt"

    noise_generator = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    direction_generator = torch.Generator(device="cpu").manual_seed(args.direction_seed)
    noise = torch.randn(
        (args.num_images, 3, args.img_size, args.img_size),
        generator=noise_generator,
        dtype=torch.float32,
    )
    fixed_direction = torch.randn(
        noise.shape,
        generator=direction_generator,
        dtype=torch.float32,
    )
    labels = torch.arange(args.num_images, dtype=torch.long) % 1000

    torch.save(noise, noise_path)
    torch.save(labels, labels_path)
    torch.save(fixed_direction, direction_path)
    torch.save(
        {
            "format_version": 1,
            "noise": noise,
            "labels": labels,
            "fixed_direction": fixed_direction,
        },
        bundle_path,
    )

    manifest = {
        "format_version": 1,
        "template_name": f"imagenet256_shared_{args.num_images}",
        "num_images": args.num_images,
        "img_size": args.img_size,
        "num_classes": 1000,
        "label_policy": "labels[i] = i % 1000",
        "noise_seed": args.noise_seed,
        "direction_seed": args.direction_seed,
        "noise": {
            "path": str(noise_path),
            "shape": list(noise.shape),
            "dtype": str(noise.dtype),
            "sha256": sha256(noise_path),
        },
        "labels": {
            "path": str(labels_path),
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
            "sha256": sha256(labels_path),
        },
        "fixed_direction": {
            "path": str(direction_path),
            "shape": list(fixed_direction.shape),
            "dtype": str(fixed_direction.dtype),
            "sha256": sha256(direction_path),
        },
        "bundle": {
            "path": str(bundle_path),
            "sha256": sha256(bundle_path),
        },
        "randomness_policy": {
            "initial_noise": "loaded from initial_noise.pt; never regenerated per method",
            "labels": "loaded from labels.pt; never regenerated per method",
            "randdir": "one fixed Gaussian direction per image, reused at every active timestep",
            "fds_mc": "uses seed_delta and seed_eps from run config",
            "fds_gd": "uses seed_eps from run config; internal eps is not regenerated from global RNG",
        },
    }
    manifest_path = args.template_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared template: {args.template_dir}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


def load_model(model_args, checkpoint_dir: Path, device: torch.device):
    model = DenoiserCustom(model_args).to(device)
    checkpoint_path = checkpoint_dir / "checkpoint-last.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    ema_state_dict1 = checkpoint["model_ema1"]
    ema_state_dict2 = checkpoint["model_ema2"]
    model.ema_params1 = [
        ema_state_dict1[name].to(device) for name, _ in model.named_parameters()
    ]
    model.ema_params2 = [
        ema_state_dict2[name].to(device) for name, _ in model.named_parameters()
    ]
    ema_state_dict = copy.deepcopy(model.state_dict())
    for i, (name, _) in enumerate(model.named_parameters()):
        ema_state_dict[name] = model.ema_params1[i]
    model.load_state_dict(ema_state_dict)
    model.eval()
    del checkpoint
    return model


def save_images(samples: torch.Tensor, output_dir: Path, start_idx: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = ((samples.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    for offset, sample in enumerate(samples):
        array = (sample.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(
            output_dir / f"{start_idx + offset:05d}.png"
        )


def run_method(args):
    bundle = torch.load(
        args.template_dir / "shared_inputs.pt",
        map_location="cpu",
        weights_only=False,
    )
    noise = bundle["noise"]
    labels = bundle["labels"]
    fixed_direction = bundle.get("fixed_direction")
    if noise.shape != (args.num_images, 3, args.img_size, args.img_size):
        raise ValueError(f"unexpected noise shape: {tuple(noise.shape)}")
    if labels.shape != (args.num_images,):
        raise ValueError(f"unexpected label shape: {tuple(labels.shape)}")
    expected_labels = torch.arange(args.num_images, dtype=torch.long) % 1000
    if not torch.equal(labels, expected_labels):
        raise ValueError("template labels are not the canonical labels[i] = i % 1000 sequence")
    if args.method == "heun_randdir":
        if fixed_direction is None:
            raise ValueError("template does not contain fixed_direction")
        if fixed_direction.shape != noise.shape:
            raise ValueError(
                f"unexpected fixed direction shape: {tuple(fixed_direction.shape)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args.output_dir / args.method
    image_dir = method_dir / "images"
    if image_dir.exists() and any(image_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{image_dir} is non-empty; use --overwrite to replace this method output"
        )
    image_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    model_args = build_args(args.method, str(args.output_dir), args.batch_size, args)
    model = load_model(model_args, args.checkpoint_dir, device)

    method_config = {
        "method": args.method,
        "template_dir": str(args.template_dir),
        "template_manifest_sha256": sha256(args.template_dir / "manifest.json"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_path": str(args.checkpoint_dir / "checkpoint-last.pth"),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "direction_seed": args.direction_seed,
        "FDS_DISABLE_TORCH_COMPILE": os.environ.get("FDS_DISABLE_TORCH_COMPILE"),
        "model_args": vars(model_args),
    }
    (method_dir / "config.json").write_text(json.dumps(method_config, indent=2) + "\n")

    direction = fixed_direction.to(device) if args.method == "heun_randdir" else None
    # gradient_stepper temporarily re-enables autograd for FDS-GD.
    with torch.no_grad():
        for start_idx in tqdm(
            range(0, args.num_images, args.batch_size),
            desc=args.method,
        ):
            end_idx = min(start_idx + args.batch_size, args.num_images)
            batch_noise = noise[start_idx:end_idx].to(device)
            batch_labels = labels[start_idx:end_idx].to(device)
            batch_direction = (
                direction[start_idx:end_idx] if direction is not None else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                samples = model.generate(
                    batch_labels,
                    model_args,
                    noise=batch_noise,
                    fixed_direction=batch_direction,
                )
            save_images(samples, image_dir, start_idx)

    print(f"Finished {args.method}: {image_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=DEFAULT_METHODS)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--noise-seed", type=int, default=20260921)
    parser.add_argument("--direction-seed", type=int, default=424242)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=1.0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--perturb-scale", type=float, default=0.01)
    parser.add_argument(
        "--perturb-schedule",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--stop-t", type=float, default=0.5)
    parser.add_argument("--seed-delta", type=int, default=42)
    parser.add_argument("--seed-eps", type=int, default=1234)
    parser.add_argument("--num-delta", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prepare:
        prepare_template(args)
        return
    if args.method is None:
        raise SystemExit("--method is required unless --prepare is used")
    run_method(args)


if __name__ == "__main__":
    main()
