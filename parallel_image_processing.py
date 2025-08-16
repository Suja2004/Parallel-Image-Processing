#!/usr/bin/env python3
"""
Parallel Image Processing (CPU, tile-based) - FIXED VERSION

Features
- Runs classic filters (gaussian blur, median, sobel, canny, sharpen) in parallel
- Splits image into tiles with halo/overlap so edges are correct
- Compares sequential vs parallel runtime
- Saves output image

Usage
------
python parallel_image_processing.py \
  --input input.jpg \
  --output out.jpg \
  --op canny \
  --tiles 2x4 \
  --grayscale

Supported ops: gaussian, median, sobel, canny, sharpen

Notes
-----
- Uses ProcessPoolExecutor (multi-core CPU). No GPU dependencies required.
- For very small images, parallel may be slower due to overhead.
- Tune --tiles and --halo for best results.
"""

from __future__ import annotations
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple
import time

import cv2
import numpy as np


# ---------------------------
# Utility dataclasses
# ---------------------------
@dataclass
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    # crop area inside the halo-expanded ROI to keep
    crop: Tuple[int, int, int, int]  # (cx0, cy0, cx1, cy1) relative to ROI


# ---------------------------
# Operations
# ---------------------------

def apply_op(img: np.ndarray, op: str, **kwargs) -> np.ndarray:
    op = op.lower()
    if op == 'gaussian':
        k = int(kwargs.get('ksize', 5))
        k = k if k % 2 == 1 else k + 1
        sigma = float(kwargs.get('sigma', 1.0))
        return cv2.GaussianBlur(img, (k, k), sigma)
    elif op == 'median':
        k = int(kwargs.get('ksize', 5))
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(img, k)
    elif op == 'sobel':
        # Apply Sobel on grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(dx, dy)
        mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        return mag
    elif op == 'canny':
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        t1 = int(kwargs.get('t1', 100))
        t2 = int(kwargs.get('t2', 200))
        return cv2.Canny(gray, t1, t2)
    elif op == 'sharpen':
        # Simple unsharp mask style kernel
        k = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, k)
    else:
        raise ValueError(f"Unsupported op: {op}")


# ---------------------------
# Tiling helpers
# ---------------------------

def make_tiles(h: int, w: int, tiles_y: int, tiles_x: int, halo: int) -> List[Tile]:
    tiles: List[Tile] = []
    ys = [round(i * h / tiles_y) for i in range(tiles_y + 1)]
    xs = [round(i * w / tiles_x) for i in range(tiles_x + 1)]

    for j in range(tiles_y):
        for i in range(tiles_x):
            y0, y1 = ys[j], ys[j + 1]
            x0, x1 = xs[i], xs[i + 1]
            # Expand ROI by halo while clamping to image bounds
            ry0 = max(0, y0 - halo)
            rx0 = max(0, x0 - halo)
            ry1 = min(h, y1 + halo)
            rx1 = min(w, x1 + halo)
            # crop region relative to expanded ROI
            cy0 = y0 - ry0
            cx0 = x0 - rx0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            tiles.append(Tile(rx0, ry0, rx1, ry1, (cx0, cy0, cx1, cy1)))
    return tiles


def _process_tile(args):
    roi, op, kwargs = args
    out = apply_op(roi, op, **kwargs)
    return out


# ---------------------------
# Main parallel runner
# ---------------------------

def run_parallel(img: np.ndarray, op: str, tiles: Tuple[int, int], halo: int, workers: int | None, **kwargs) -> np.ndarray:
    h, w = img.shape[:2]
    tiles_y, tiles_x = tiles
    tile_defs = make_tiles(h, w, tiles_y, tiles_x, halo)

    # Package tasks
    tasks = []
    rois = []
    for t in tile_defs:
        roi = img[t.y0:t.y1, t.x0:t.x1]
        rois.append(roi)
        tasks.append((roi, op, kwargs))

    # Execute in parallel
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_tile, arg) for arg in tasks]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - start

    # Create output array with proper shape and dtype
    # Get a sample result to determine output characteristics
    sample_result = results[0]
    if sample_result.ndim == 2:
        out = np.zeros((h, w), dtype=sample_result.dtype)
    else:
        out = np.zeros((h, w, sample_result.shape[2]), dtype=sample_result.dtype)

    # Stitch results back together
    for idx, t in enumerate(tile_defs):
        res = results[idx]
        cx0, cy0, cx1, cy1 = t.crop
        
        # Extract the region we want from the processed tile
        if res.ndim == 2:
            cropped = res[cy0:cy1, cx0:cx1]
        else:
            cropped = res[cy0:cy1, cx0:cx1, :]
        
        # Calculate the actual target region in the output
        target_y0 = t.y0
        target_y1 = target_y0 + (cy1 - cy0)
        target_x0 = t.x0  
        target_x1 = target_x0 + (cx1 - cx0)
        
        # Make sure we don't go out of bounds
        target_y1 = min(target_y1, h)
        target_x1 = min(target_x1, w)
        
        # Adjust cropped region if needed
        actual_h = target_y1 - target_y0
        actual_w = target_x1 - target_x0
        
        if cropped.ndim == 2:
            cropped = cropped[:actual_h, :actual_w]
            out[target_y0:target_y1, target_x0:target_x1] = cropped
        else:
            cropped = cropped[:actual_h, :actual_w, :]
            out[target_y0:target_y1, target_x0:target_x1, :] = cropped

    return out, elapsed


def run_sequential(img: np.ndarray, op: str, **kwargs) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = apply_op(img, op, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed


# ---------------------------
# CLI
# ---------------------------

def parse_tiles(s: str) -> Tuple[int, int]:
    if 'x' not in s.lower():
        raise argparse.ArgumentTypeError("Tiles must be in AxB format, e.g., 2x4")
    a, b = s.lower().split('x')
    ty, tx = int(a), int(b)
    if ty < 1 or tx < 1:
        raise argparse.ArgumentTypeError("Tiles must be >= 1x1")
    return ty, tx


def main():
    p = argparse.ArgumentParser(description="Parallel Image Processing (CPU, tile-based)")
    p.add_argument('--input', required=True, help='Path to input image')
    p.add_argument('--output', required=True, help='Path to save output image')
    p.add_argument('--op', default='canny', choices=['gaussian', 'median', 'sobel', 'canny', 'sharpen'])
    p.add_argument('--tiles', type=parse_tiles, default=(2, 2), help='Tiles as AxB, e.g., 2x4 (A=rows, B=cols)')
    p.add_argument('--halo', type=int, default=8, help='Overlap (pixels) added around each tile')
    p.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: os.cpu_count())')
    p.add_argument('--grayscale', action='store_true', help='Force grayscale before processing (except ops that convert internally)')
    # op-specific knobs
    p.add_argument('--ksize', type=int, default=5, help='Kernel size for gaussian/median')
    p.add_argument('--sigma', type=float, default=1.0, help='Sigma for gaussian')
    p.add_argument('--t1', type=int, default=100, help='Canny threshold1')
    p.add_argument('--t2', type=int, default=200, help='Canny threshold2')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("Failed to load image. Unsupported format or corrupt file.")

    if args.grayscale and args.op not in ('canny', 'sobel'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Input image shape: {img.shape}")

    # Sequential reference
    seq_out, t_seq = run_sequential(img, args.op, ksize=args.ksize, sigma=args.sigma, t1=args.t1, t2=args.t2)
    print(f"Sequential output shape: {seq_out.shape}")

    # Parallel
    par_out, t_par = run_parallel(img, args.op, args.tiles, args.halo, args.workers, ksize=args.ksize, sigma=args.sigma, t1=args.t1, t2=args.t2)
    print(f"Parallel output shape: {par_out.shape}")

    # Prefer parallel output to save (identical visually except for floating rounding)
    out = par_out

    # If single-channel, save as PNG to preserve 8-bit grayscale; for multi-channel keep as BGR
    if out.ndim == 2:
        save_ok = cv2.imwrite(args.output, out)
    else:
        save_ok = cv2.imwrite(args.output, out)

    if not save_ok:
        raise SystemExit("Failed to save output image.")

    speedup = t_seq / t_par if t_par > 0 else float('inf')
    print(f"Operation      : {args.op}")
    print(f"Image          : {args.input} -> {args.output}")
    print(f"Tiles (YxX)    : {args.tiles[0]}x{args.tiles[1]} | Halo: {args.halo} | Workers: {args.workers or os.cpu_count()}")
    print(f"Sequential time: {t_seq*1000:.2f} ms")
    print(f"Parallel time  : {t_par*1000:.2f} ms")
    print(f"Speedup        : {speedup:.2f}x")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # running in Colab/Notebook with no CLI args
        args = argparse.Namespace(
            input="sample.jpg",
            output="out.jpg",
            op="canny",
            tiles=(2, 2),
            halo=8,
            workers=None,
            grayscale=False,
            ksize=5,
            sigma=1.0,
            t1=100,
            t2=200,
        )
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        out, _ = run_parallel(img, args.op, args.tiles, args.halo, args.workers,
                              ksize=args.ksize, sigma=args.sigma, t1=args.t1, t2=args.t2)
        cv2.imwrite(args.output, out)
        print(f"Saved {args.output}")
    else:
        main()