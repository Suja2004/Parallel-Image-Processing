#!/usr/bin/env python3
"""
Enhanced Multi-Result Image Processing Backend
==============================================

Features:
- Process single image with multiple filters simultaneously
- Batch process multiple images with multiple filters
- Compare results side-by-side
- Filter combinations and presets
- Advanced parameter variations

New Endpoints:
- POST /api/process-multi - Apply multiple filters to one image
- POST /api/batch-process - Process multiple images with multiple filters
- GET /api/presets - Get filter presets and combinations
- POST /api/compare - Generate comparison grids
"""

import os
import time
import uuid
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import hashlib

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, NotFound
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class Config:
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    MAX_FILE_SIZE = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    CLEANUP_INTERVAL = 3600
    FILE_RETENTION = 24 * 3600
    MAX_CONCURRENT_TASKS = 8
    MAX_BATCH_SIZE = 10
    MAX_FILTERS_PER_IMAGE = 8


app.config.from_object(Config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

tasks: Dict[str, Dict] = {}
task_executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_TASKS)


@dataclass
class BatchTask:
    task_id: str
    status: str
    batch_type: str
    input_files: List[str]
    operations: List[Dict]
    output_files: List[str] = None
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: int = 0
    results_summary: Optional[Dict] = None


# Filter presets and combinations
FILTER_PRESETS = {
    'edge_detection_suite': [
        {'operation': 'canny', 'parameters': {
            't1': 50, 't2': 150}, 'name': 'Canny (Soft)'},
        {'operation': 'canny', 'parameters': {
            't1': 100, 't2': 200}, 'name': 'Canny (Standard)'},
        {'operation': 'canny', 'parameters': {
            't1': 150, 't2': 250}, 'name': 'Canny (Sharp)'},
        {'operation': 'sobel', 'parameters': {}, 'name': 'Sobel'}
    ],
    'blur_variations': [
        {'operation': 'gaussian', 'parameters': {
            'ksize': 3, 'sigma': 0.5}, 'name': 'Light Blur'},
        {'operation': 'gaussian', 'parameters': {
            'ksize': 5, 'sigma': 1.0}, 'name': 'Medium Blur'},
        {'operation': 'gaussian', 'parameters': {
            'ksize': 7, 'sigma': 1.5}, 'name': 'Heavy Blur'},
        {'operation': 'median', 'parameters': {
            'ksize': 5}, 'name': 'Median Filter'}
    ],
    'enhancement_pack': [
        {'operation': 'sharpen', 'parameters': {}, 'name': 'Sharpen'},
        {'operation': 'gaussian', 'parameters': {
            'ksize': 3, 'sigma': 0.5}, 'name': 'Light Smooth'},
        {'operation': 'median', 'parameters': {
            'ksize': 3}, 'name': 'Noise Reduction'}
    ],
    'comparison_set': [
        {'operation': 'canny', 'parameters': {
            't1': 100, 't2': 200}, 'name': 'Canny'},
        {'operation': 'sobel', 'parameters': {}, 'name': 'Sobel'},
        {'operation': 'gaussian', 'parameters': {
            'ksize': 5, 'sigma': 1.0}, 'name': 'Gaussian'},
        {'operation': 'sharpen', 'parameters': {}, 'name': 'Sharpen'}
    ]
}

# Core image processing functions 


@dataclass
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    crop: Tuple[int, int, int, int]


def apply_operation(img: np.ndarray, operation: str, **params) -> np.ndarray:
    operation = operation.lower()

    if operation == 'gaussian':
        ksize = int(params.get('ksize', 5))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        sigma = float(params.get('sigma', 1.0))
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif operation == 'median':
        ksize = int(params.get('ksize', 5))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(img, ksize)
    elif operation == 'sobel':
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        magnitude = np.clip(magnitude / (magnitude.max() + 1e-6)
                            * 255, 0, 255).astype(np.uint8)
        return magnitude
    elif operation == 'canny':
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        t1 = int(params.get('t1', 100))
        t2 = int(params.get('t2', 200))
        return cv2.Canny(gray, t1, t2)
    elif operation == 'sharpen':
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def create_tiles(h: int, w: int, tiles_y: int, tiles_x: int, halo: int) -> List[Tile]:
    tiles = []
    ys = [round(i * h / tiles_y) for i in range(tiles_y + 1)]
    xs = [round(i * w / tiles_x) for i in range(tiles_x + 1)]

    for j in range(tiles_y):
        for i in range(tiles_x):
            y0, y1 = ys[j], ys[j + 1]
            x0, x1 = xs[i], xs[i + 1]

            ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
            ry1, rx1 = min(h, y1 + halo), min(w, x1 + halo)

            cy0, cx0 = y0 - ry0, x0 - rx0
            cy1, cx1 = cy0 + (y1 - y0), cx0 + (x1 - x0)

            tiles.append(Tile(rx0, ry0, rx1, ry1, (cx0, cy0, cx1, cy1)))

    return tiles


class ImageProcessor:
    def __init__(self, img: np.ndarray, operation: str, params: dict):
        self.img = img
        self.operation = operation
        self.params = params

    def process_tile(self, tile: Tile) -> Tuple[np.ndarray, Tile]:
        roi = self.img[tile.y0:tile.y1, tile.x0:tile.x1]
        result = apply_operation(roi, self.operation, **self.params)
        return result, tile


def process_single_operation(img: np.ndarray, operation: str, params: dict) -> Tuple[np.ndarray, dict]:
    """Process image with a single operation"""
    h, w = img.shape[:2]
    image_size = h * w

    if image_size < 1_000_000:
        start_time = time.perf_counter()
        result = apply_operation(img, operation, **params)
        processing_time = time.perf_counter() - start_time

        metrics = {
            'method': 'sequential',
            'processing_time_ms': processing_time * 1000,
            'image_size_mp': image_size / 1_000_000
        }
    else:
        # Parallel processing logic
        if operation in ['canny', 'sobel']:
            tiles_config, workers, halo = (4, 4), 4, 8
        elif operation == 'median':
            tiles_config, workers, halo = (
                2, 2), 2, max(8, params.get('ksize', 5))
        else:
            tiles_config, workers, halo = (3, 3), 4, max(
                8, params.get('ksize', 5) * 2)

        tiles_y, tiles_x = tiles_config
        tile_defs = create_tiles(h, w, tiles_y, tiles_x, halo)
        processor = ImageProcessor(img, operation, params)

        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(processor.process_tile, tile)
                       for tile in tile_defs]
            results = [future.result() for future in futures]
        processing_time = time.perf_counter() - start_time

        sample_result = results[0][0]
        if sample_result.ndim == 2:
            output = np.zeros((h, w), dtype=sample_result.dtype)
        else:
            output = np.zeros(
                (h, w, sample_result.shape[2]), dtype=sample_result.dtype)

        for result, tile in results:
            cx0, cy0, cx1, cy1 = tile.crop
            if result.ndim == 2:
                cropped = result[cy0:cy1, cx0:cx1]
            else:
                cropped = result[cy0:cy1, cx0:cx1, :]

            target_y0, target_x0 = tile.y0, tile.x0
            target_y1, target_x1 = target_y0 + \
                (cy1 - cy0), target_x0 + (cx1 - cx0)
            target_y1, target_x1 = min(target_y1, h), min(target_x1, w)
            actual_h, actual_w = target_y1 - target_y0, target_x1 - target_x0

            if cropped.ndim == 2:
                output[target_y0:target_y1,
                       target_x0:target_x1] = cropped[:actual_h, :actual_w]
            else:
                output[target_y0:target_y1, target_x0:target_x1,
                       :] = cropped[:actual_h, :actual_w, :]

        result = output
        metrics = {
            'method': 'parallel_tiled',
            'processing_time_ms': processing_time * 1000,
            'tiles': f"{tiles_y}x{tiles_x}",
            'workers': workers,
            'halo': halo,
            'image_size_mp': image_size / 1_000_000
        }

    return result, metrics

# Batch processing functions


def process_multi_filter_task(task_id: str):
    """Process single image with multiple filters"""
    task = tasks[task_id]
    try:
        task['status'] = 'processing'
        task['start_time'] = time.time()

        input_file = task['input_files'][0]
        operations = task['operations']

        # Load image
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image")

        output_files = []
        all_metrics = []

        # Process with each filter
        for i, op_config in enumerate(operations):
            operation = op_config['operation']
            parameters = op_config.get('parameters', {})
            name = op_config.get('name', f"{operation}_{i}")

            # Update progress
            progress = 20 + int((i / len(operations)) * 60)
            task['progress'] = progress

            # Process image
            result_img, metrics = process_single_operation(
                img, operation, parameters)

            # Save result
            safe_name = "".join(c for c in name if c.isalnum()
                                or c in (' ', '-', '_')).rstrip()
            output_filename = f"multi_{task_id}_{safe_name}.png"
            output_path = os.path.join(
                app.config['PROCESSED_FOLDER'], output_filename)
            cv2.imwrite(output_path, result_img)

            output_files.append(output_filename)
            metrics['filter_name'] = name
            metrics['operation'] = operation
            all_metrics.append(metrics)

        task['status'] = 'completed'
        task['output_files'] = output_files
        task['end_time'] = time.time()
        task['progress'] = 100
        task['results_summary'] = {
            'total_filters': len(operations),
            'total_time_ms': sum(m['processing_time_ms'] for m in all_metrics),
            'metrics_per_filter': all_metrics
        }

        logger.info(f"Multi-filter task {task_id} completed")

    except Exception as e:
        task['status'] = 'failed'
        task['error_message'] = str(e)
        task['end_time'] = time.time()
        logger.error(f"Multi-filter task {task_id} failed: {str(e)}")


def process_batch_images_task(task_id: str):
    """Process multiple images with multiple filters"""
    task = tasks[task_id]
    try:
        task['status'] = 'processing'
        task['start_time'] = time.time()

        input_files = task['input_files']
        operations = task['operations']

        output_files = []
        all_results = []
        total_operations = len(input_files) * len(operations)
        completed_operations = 0

        # Process each image with each filter
        for img_idx, input_file in enumerate(input_files):
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file)
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to load {input_file}")
                continue

            img_results = {'input_file': input_file, 'results': []}

            for op_idx, op_config in enumerate(operations):
                operation = op_config['operation']
                parameters = op_config.get('parameters', {})
                name = op_config.get('name', f"{operation}_{op_idx}")

                progress = 10 + \
                    int((completed_operations / total_operations) * 80)
                task['progress'] = progress

                try:
                    result_img, metrics = process_single_operation(
                        img, operation, parameters)

                    base_name = os.path.splitext(input_file)[0]
                    safe_name = "".join(
                        c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    output_filename = f"batch_{task_id}_{base_name}_{safe_name}.png"
                    output_path = os.path.join(
                        app.config['PROCESSED_FOLDER'], output_filename)
                    cv2.imwrite(output_path, result_img)

                    output_files.append(output_filename)
                    metrics['filter_name'] = name
                    metrics['operation'] = operation
                    metrics['input_file'] = input_file
                    img_results['results'].append({
                        'output_file': output_filename,
                        'metrics': metrics
                    })

                except Exception as e:
                    logger.error(
                        f"Failed to process {input_file} with {operation}: {str(e)}")

                completed_operations += 1

            all_results.append(img_results)

        task['status'] = 'completed'
        task['output_files'] = output_files
        task['end_time'] = time.time()
        task['progress'] = 100
        task['results_summary'] = {
            'total_images': len(input_files),
            'total_filters': len(operations),
            'total_outputs': len(output_files),
            'results_by_image': all_results
        }

        logger.info(f"Batch task {task_id} completed")

    except Exception as e:
        task['status'] = 'failed'
        task['error_message'] = str(e)
        task['end_time'] = time.time()
        logger.error(f"Batch task {task_id} failed: {str(e)}")

# API Routes


@app.route('/', methods=['GET'])
def api_info():
    """Base API route - provides information about available endpoints"""
    return jsonify({
        'name': 'Enhanced Multi-Result Image Processing Backend',
        'version': '2.0',
        'description': 'A Flask backend for processing images with multiple filters, batch processing, and comparison tools',
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat(),
        'endpoints': {
            'health': {
                'method': 'GET',
                'url': '/api/health',
                'description': 'Check server health and status'
            },
            'upload': {
                'method': 'POST',
                'url': '/api/upload',
                'description': 'Upload single or multiple image files',
                'accepts': 'multipart/form-data',
                'max_files': app.config['MAX_BATCH_SIZE'],
                'max_file_size': f"{app.config['MAX_FILE_SIZE'] // (1024*1024)}MB",
                'supported_formats': list(app.config['ALLOWED_EXTENSIONS'])
            },
            'download': {
                'method': 'GET',
                'url': '/api/download/<filename>',
                'description': 'Download processed image files'
            },
            'presets': {
                'method': 'GET',
                'url': '/api/presets',
                'description': 'Get available filter presets and combinations'
            },
            'process_multi': {
                'method': 'POST',
                'url': '/api/process-multi',
                'description': 'Apply multiple filters to a single image',
                'accepts': 'application/json',
                'parameters': {
                    'filename': 'string (required) - uploaded file name',
                    'preset': 'string (optional) - preset name from /api/presets',
                    'filters': 'array (optional) - custom filter configurations'
                },
                'example': {
                    'filename': 'uploaded_image.jpg',
                    'preset': 'edge_detection_suite'
                }
            },
            'batch_process': {
                'method': 'POST',
                'url': '/api/batch-process',
                'description': 'Process multiple images with multiple filters',
                'accepts': 'application/json',
                'parameters': {
                    'filenames': 'array (required) - list of uploaded file names',
                    'preset': 'string (optional) - preset name',
                    'filters': 'array (optional) - custom filter configurations'
                },
                'example': {
                    'filenames': ['image1.jpg', 'image2.jpg'],
                    'preset': 'comparison_set'
                }
            },
            'generate_variations': {
                'method': 'POST',
                'url': '/api/generate-variations',
                'description': 'Generate parameter variations of a single filter',
                'accepts': 'application/json',
                'parameters': {
                    'filename': 'string (required) - uploaded file name',
                    'operation': 'string (required) - filter operation name',
                    'variations': 'object (required) - parameter variations'
                },
                'example': {
                    'filename': 'test.jpg',
                    'operation': 'gaussian',
                    'variations': {
                        'ksize': [3, 5, 7],
                        'sigma': [0.5, 1.0, 1.5]
                    }
                }
            },
            'status': {
                'method': 'GET',
                'url': '/api/status/<task_id>',
                'description': 'Check processing task status and get results'
            }
        },
        'supported_operations': {
            'gaussian': {
                'description': 'Gaussian blur filter',
                'parameters': {
                    'ksize': 'int (odd number) - kernel size, default: 5',
                    'sigma': 'float - standard deviation, default: 1.0'
                }
            },
            'median': {
                'description': 'Median blur filter for noise reduction',
                'parameters': {
                    'ksize': 'int (odd number) - kernel size, default: 5'
                }
            },
            'canny': {
                'description': 'Canny edge detection',
                'parameters': {
                    't1': 'int - low threshold, default: 100',
                    't2': 'int - high threshold, default: 200'
                }
            },
            'sobel': {
                'description': 'Sobel edge detection',
                'parameters': {}
            },
            'sharpen': {
                'description': 'Sharpen filter to enhance edges',
                'parameters': {}
            }
        },
        'available_presets': list(FILTER_PRESETS.keys()),
        'limits': {
            'max_batch_size': app.config['MAX_BATCH_SIZE'],
            'max_filters_per_image': app.config['MAX_FILTERS_PER_IMAGE'],
            'max_file_size_mb': app.config['MAX_FILE_SIZE'] // (1024*1024),
            'max_concurrent_tasks': app.config['MAX_CONCURRENT_TASKS']
        },
        'quick_start': {
            '1': 'Upload an image: POST /api/upload with file in form-data',
            '2': 'Get available presets: GET /api/presets',
            '3': 'Process with preset: POST /api/process-multi with filename and preset',
            '4': 'Check status: GET /api/status/<task_id>',
            '5': 'Download results: GET /api/download/<filename>'
        }
    })


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get available filter presets"""
    return jsonify(FILTER_PRESETS)


@app.route('/api/process-multi', methods=['POST'])
def process_multi_filters():
    """Apply multiple filters to single image"""
    data = request.get_json()

    if not data or 'filename' not in data:
        return jsonify({'error': 'Missing filename'}), 400

    filename = data['filename']

    if 'preset' in data:
        preset_name = data['preset']
        if preset_name not in FILTER_PRESETS:
            return jsonify({'error': f'Unknown preset: {preset_name}'}), 400
        operations = FILTER_PRESETS[preset_name]
    elif 'filters' in data:
        operations = data['filters']
        if len(operations) > app.config['MAX_FILTERS_PER_IMAGE']:
            return jsonify({'error': f'Maximum {app.config["MAX_FILTERS_PER_IMAGE"]} filters allowed'}), 400
    else:
        return jsonify({'error': 'Either preset or filters must be specified'}), 400

    # Validate file exists
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    # Create task
    task_id = generate_task_id()
    task = BatchTask(
        task_id=task_id,
        status='pending',
        batch_type='multi_filter',
        input_files=[filename],
        operations=operations
    )

    tasks[task_id] = asdict(task)
    task_executor.submit(process_multi_filter_task, task_id)

    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'batch_type': 'multi_filter',
        'filters_count': len(operations)
    })


@app.route('/api/batch-process', methods=['POST'])
def batch_process_images():
    """Process multiple images with multiple filters"""
    data = request.get_json()

    if not data or 'filenames' not in data:
        return jsonify({'error': 'Missing filenames'}), 400

    filenames = data['filenames']
    if len(filenames) > app.config['MAX_BATCH_SIZE']:
        return jsonify({'error': f'Maximum {app.config["MAX_BATCH_SIZE"]} images allowed'}), 400

    # Get filters
    if 'preset' in data:
        preset_name = data['preset']
        if preset_name not in FILTER_PRESETS:
            return jsonify({'error': f'Unknown preset: {preset_name}'}), 400
        operations = FILTER_PRESETS[preset_name]
    elif 'filters' in data:
        operations = data['filters']
    else:
        return jsonify({'error': 'Either preset or filters must be specified'}), 400

    # Validate files exist
    for filename in filenames:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404

    # Create task
    task_id = generate_task_id()
    task = BatchTask(
        task_id=task_id,
        status='pending',
        batch_type='multi_image',
        input_files=filenames,
        operations=operations
    )

    tasks[task_id] = asdict(task)
    task_executor.submit(process_batch_images_task, task_id)

    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'batch_type': 'multi_image',
        'images_count': len(filenames),
        'filters_count': len(operations),
        'total_outputs': len(filenames) * len(operations)
    })


@app.route('/api/generate-variations', methods=['POST'])
def generate_parameter_variations():
    """Generate multiple variations of the same filter with different parameters"""
    data = request.get_json()

    filename = data.get('filename')
    operation = data.get('operation')
    variations = data.get('variations', {})

    if not all([filename, operation, variations]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Generate parameter combinations
    operations = []
    param_names = list(variations.keys())
    param_values = list(variations.values())

    for i, combination in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, combination))
        param_str = '_'.join([f"{k}{v}" for k, v in params.items()])
        operations.append({
            'operation': operation,
            'parameters': params,
            'name': f"{operation}_{param_str}"
        })

        if len(operations) >= app.config['MAX_FILTERS_PER_IMAGE']:
            break

    # Create and submit task
    task_id = generate_task_id()
    task = BatchTask(
        task_id=task_id,
        status='pending',
        batch_type='variations',
        input_files=[filename],
        operations=operations
    )

    tasks[task_id] = asdict(task)
    task_executor.submit(process_multi_filter_task, task_id)

    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'batch_type': 'variations',
        'variations_count': len(operations)
    })

# Enhanced status endpoint for batch results


@app.route('/api/status/<task_id>', methods=['GET'])
def get_batch_status(task_id):
    """Get batch task status with detailed results"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    response = {
        'task_id': task_id,
        'status': task['status'],
        'progress': task.get('progress', 0),
        'batch_type': task.get('batch_type', 'single')
    }

    if task['status'] == 'completed':
        response.update({
            'output_files': task.get('output_files', []),
            'results_summary': task.get('results_summary', {}),
            'processing_time': task['end_time'] - task['start_time'] if task.get('start_time') else None
        })
    elif task['status'] == 'failed':
        response['error_message'] = task.get('error_message')

    return jsonify(response)

# Utility functions


def generate_task_id():
    return str(uuid.uuid4())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Include all previous endpoints (health, upload, download, etc.)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'active_tasks': len([t for t in tasks.values() if t['status'] in ['pending', 'processing']]),
        'batch_capabilities': {
            'max_batch_size': app.config['MAX_BATCH_SIZE'],
            'max_filters_per_image': app.config['MAX_FILTERS_PER_IMAGE'],
            'available_presets': list(FILTER_PRESETS.keys())
        }
    })


@app.route('/api/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload image file (supports multiple files)"""
    if 'files' in request.files:
        files = request.files.getlist('files')
    elif 'file' in request.files:
        files = [request.files['file']]
    else:
        return jsonify({'error': 'No files provided'}), 400

    if len(files) > app.config['MAX_BATCH_SIZE']:
        return jsonify({'error': f'Maximum {app.config["MAX_BATCH_SIZE"]} files allowed'}), 400

    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > app.config['MAX_FILE_SIZE']:
            continue

        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        file.save(filepath)

        # Get image info
        img = cv2.imread(filepath)
        height, width = img.shape[:2] if img is not None else (0, 0)

        results.append({
            'filename': unique_filename,
            'original_filename': filename,
            'size_bytes': file_size,
            'dimensions': {'width': width, 'height': height}
        })

    return jsonify({'uploaded_files': results})


@app.route('/api/preview/<filename>', methods=['GET'])
def preview_file(filename):
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


def cleanup_files():
    """Deletes files in uploads and processed folders older than FILE_RETENTION."""
    while True:
        try:
            now = datetime.now()
            for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_mtime = datetime.fromtimestamp(
                            os.path.getmtime(filepath))
                        if now - file_mtime > timedelta(minutes=3):
                            os.remove(filepath)
                            logger.info(f"Deleted old file: {filepath}")
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")

        time.sleep(60)


if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_files, daemon=True)
    cleanup_thread.start()
    logger.info("Starting Enhanced Multi-Result Image Processing Backend")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
