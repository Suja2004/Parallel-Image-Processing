#!/usr/bin/env python3
"""
Threaded Parallel Image Processing (CPU, tile-based)

Features
- Runs classic filters using ThreadPoolExecutor instead of ProcessPoolExecutor
- Much lower overhead than multiprocessing
- Better for I/O bound and moderate CPU tasks
- Splits image into tiles with halo/overlap so edges are correct

Usage
------
python threaded_image_processing.py \
  --input input.jpg \
  --output out.jpg \
  --op canny \
  --tiles 4x4

Supported ops: gaussian, median, sobel, canny, sharpen
"""

from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class TileProcessor:
    """Wrapper to avoid pickling issues and share data efficiently"""
    def __init__(self, img: np.ndarray, op: str, kwargs: dict):
        self.img = img
        self.op = op
        self.kwargs = kwargs
    
    def process_tile(self, tile: Tile) -> Tuple[np.ndarray, Tile]:
        # Extract ROI
        roi = self.img[tile.y0:tile.y1, tile.x0:tile.x1]
        # Process it
        result = apply_op(roi, self.op, **self.kwargs)
        return result, tile


# ---------------------------
# Main parallel runner
# ---------------------------

def run_threaded_parallel(img: np.ndarray, op: str, tiles: Tuple[int, int], halo: int, workers: int | None, **kwargs) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    tiles_y, tiles_x = tiles
    tile_defs = make_tiles(h, w, tiles_y, tiles_x, halo)
    
    # Create processor with shared data
    processor = TileProcessor(img, op, kwargs)
    
    # Execute in parallel using threads
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = [executor.submit(processor.process_tile, tile) for tile in tile_defs]
        # Collect results
        results = []
        for future in as_completed(futures):
            result, tile = future.result()
            results.append((result, tile))
    elapsed = time.perf_counter() - start
    
    # Sort results by original tile order to maintain consistency
    tile_to_index = {id(tile): i for i, tile in enumerate(tile_defs)}
    results.sort(key=lambda x: tile_to_index[id(x[1])])
    
    # Create output array with proper shape and dtype
    sample_result = results[0][0]
    if sample_result.ndim == 2:
        out = np.zeros((h, w), dtype=sample_result.dtype)
    else:
        out = np.zeros((h, w, sample_result.shape[2]), dtype=sample_result.dtype)
    
    # Stitch results back together
    for result, tile in results:
        cx0, cy0, cx1, cy1 = tile.crop
        
        # Extract the region we want from the processed tile
        if result.ndim == 2:
            cropped = result[cy0:cy1, cx0:cx1]
        else:
            cropped = result[cy0:cy1, cx0:cx1, :]
        
        # Calculate target region in output
        target_y0 = tile.y0
        target_y1 = target_y0 + (cy1 - cy0)
        target_x0 = tile.x0
        target_x1 = target_x0 + (cx1 - cx0)
        
        # Bounds checking
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
    p = argparse.ArgumentParser(description="Threaded Parallel Image Processing")
    p.add_argument('--input', required=True, help='Path to input image')
    p.add_argument('--output', required=True, help='Path to save output image')
    p.add_argument('--op', default='canny', choices=['gaussian', 'median', 'sobel', 'canny', 'sharpen'])
    p.add_argument('--tiles', type=parse_tiles, default=(2, 2), help='Tiles as AxB, e.g., 2x4')
    p.add_argument('--halo', type=int, default=8, help='Overlap pixels around each tile')
    p.add_argument('--workers', type=int, default=None, help='Number of worker threads (default: 4)')
    p.add_argument('--grayscale', action='store_true', help='Force grayscale before processing')
    # op-specific parameters
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

    # Set default workers for threading (usually 4 is good for I/O bound tasks)
    if args.workers is None:
        args.workers = 4

    # Sequential reference
    seq_out, t_seq = run_sequential(img, args.op, ksize=args.ksize, sigma=args.sigma, t1=args.t1, t2=args.t2)
    print(f"Sequential output shape: {seq_out.shape}")

    # Threaded parallel
    par_out, t_par = run_threaded_parallel(img, args.op, args.tiles, args.halo, args.workers, 
                                         ksize=args.ksize, sigma=args.sigma, t1=args.t1, t2=args.t2)
    print(f"Threaded parallel output shape: {par_out.shape}")

    # Save parallel output
    out = par_out
    save_ok = cv2.imwrite(args.output, out)
    if not save_ok:
        raise SystemExit("Failed to save output image.")

    speedup = t_seq / t_par if t_par > 0 else float('inf')
    print(f"Operation         : {args.op}")
    print(f"Image             : {args.input} -> {args.output}")
    print(f"Tiles (YxX)       : {args.tiles[0]}x{args.tiles[1]} | Halo: {args.halo} | Workers: {args.workers}")
    print(f"Sequential time   : {t_seq*1000:.2f} ms")
    print(f"Threaded time     : {t_par*1000:.2f} ms")
    print(f"Speedup           : {speedup:.2f}x")
    print(f"Threading overhead: {((t_par - t_seq)/t_seq)*100:+.1f}%")


if __name__ == "__main__":
    main()