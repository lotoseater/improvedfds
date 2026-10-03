#!/usr/bin/env python3
"""Create a canonical per-sample initial-noise array for paired runs."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-images', type=int, required=True)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=16)
    args = parser.parse_args()

    if args.num_images <= 0 or args.chunk_size <= 0:
        raise ValueError('num-images and chunk-size must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists')

    shape = (args.num_images, 3, args.img_size, args.img_size)
    noise = np.lib.format.open_memmap(
        args.output, mode='w+', dtype=np.float32, shape=shape
    )
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    for start in range(0, args.num_images, args.chunk_size):
        end = min(start + args.chunk_size, args.num_images)
        block = torch.randn(
            (end - start, 3, args.img_size, args.img_size),
            generator=generator,
            dtype=torch.float32,
        )
        noise[start:end] = block.numpy()
    noise.flush()
    print(f'Wrote {args.output}: shape={shape}, dtype=float32, seed={args.seed}')


if __name__ == '__main__':
    main()
