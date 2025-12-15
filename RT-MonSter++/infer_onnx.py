"""
RT-MonSter++ ONNX Inference Script

Usage:
    # Single image pair inference
    python infer_onnx.py --onnx_path ./rt-monster_544_960.onnx \
        -l ./demo_img/left.png -r ./demo_img/right.png \
        --output_directory ./Output/onnx_demo \
        --save_colormap \
        --save_numpy

    # Batch image inference
    python infer_onnx.py --onnx_path ./rt-monster_544_960.onnx \
        -l "/data/StereoData/kitti/2015/testing/image_2/*_10.png" \
        -r "/data/StereoData/kitti/2015/testing/image_3/*_10.png" \
        --output_directory ./Output/kitti/2015_onnx
"""

import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import skimage.io
import cv2
import time

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime-gpu or pip install onnxruntime")
    sys.exit(1)


class InputPadder:
    """Pad images to dimensions divisible by a specified number"""
    def __init__(self, dims, divis_by=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        """Pad input images (numpy arrays)"""
        outputs = []
        for x in inputs:
            # x: [B, C, H, W]
            padded = np.pad(
                x,
                ((0, 0), (0, 0), (self._pad[2], self._pad[3]), (self._pad[0], self._pad[1])),
                mode='edge'
            )
            outputs.append(padded)
        return outputs

    def unpad(self, x):
        """Remove padding (numpy array)"""
        # x: [B, C, H, W]
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_image(imfile):
    """Load image and convert to model input format"""
    img = np.array(Image.open(imfile)).astype(np.float32)
    if img.ndim == 2:
        # Convert grayscale to RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        # Convert RGBA to RGB, remove alpha channel
        img = img[..., :3]
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    # Add batch dimension
    img = img[np.newaxis, ...]
    return img


def create_onnx_session(onnx_path, use_gpu=True):
    """Create ONNX Runtime session"""
    if use_gpu:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CPUExecutionProvider']
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    
    # Print the provider being used
    print(f"Provider: {session.get_providers()}")
    
    return session


def get_model_input_size(session):
    """Get the expected input size of the model"""
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    # shape: ['batch', 3, 'height', 'width'] or [1, 3, H, W]
    if isinstance(shape[2], int) and isinstance(shape[3], int):
        return shape[2], shape[3]  # height, width
    return None, None


def preprocess_image(image, target_height=None, target_width=None):
    """
    Preprocess image to fit model input
    
    Args:
        image: numpy array [B, C, H, W]
        target_height: target height (if resize needed)
        target_width: target width (if resize needed)
    
    Returns:
        processed_image: numpy array [B, C, H, W]
        scale_factor: (scale_h, scale_w) for post-processing
    """
    _, _, h, w = image.shape
    scale_h, scale_w = 1.0, 1.0
    
    if target_height is not None and target_width is not None:
        if h != target_height or w != target_width:
            # Need to resize
            scale_h = h / target_height
            scale_w = w / target_width
            # Use cv2 resize
            img_resized = cv2.resize(
                image[0].transpose(1, 2, 0),  # CHW -> HWC
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
            image = img_resized.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> CHW, add batch
    
    return image, (scale_h, scale_w)


def postprocess_disparity(disp, original_size, scale_factor):
    """
    Post-process disparity map
    
    Args:
        disp: numpy array [B, 1, H, W]
        original_size: (H, W) original image size
        scale_factor: (scale_h, scale_w)
    
    Returns:
        disp: numpy array [H, W]
    """
    disp = disp.squeeze()  # Remove batch and channel dimensions
    
    scale_h, scale_w = scale_factor
    if scale_h != 1.0 or scale_w != 1.0:
        # Resize back to original size
        h, w = original_size
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        # Disparity needs to be scaled by width ratio
        disp = disp * scale_w
    
    return disp


def demo(args):
    """Main inference function"""
    
    # Check if ONNX model exists
    if not os.path.exists(args.onnx_path):
        print(f"Error: ONNX model file not found: {args.onnx_path}")
        sys.exit(1)
    
    # Create ONNX session
    print(f"Loading ONNX model: {args.onnx_path}")
    session = create_onnx_session(args.onnx_path, use_gpu=not args.cpu)
    
    # Get model input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Input nodes: {input_names}")
    print(f"Output nodes: {output_names}")
    
    # Get expected model input size
    model_height, model_width = get_model_input_size(session)
    if model_height is not None:
        print(f"Model input size: {model_height} x {model_width}")
    else:
        print("Model uses dynamic input size")
    
    # Create output directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Get image lists
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    
    if len(left_images) == 0:
        print(f"Error: No left images found: {args.left_imgs}")
        sys.exit(1)
    
    if len(left_images) != len(right_images):
        print(f"Warning: Left/right image count mismatch ({len(left_images)} vs {len(right_images)})")
    
    print(f"Found {len(left_images)} image pairs. Saving to {output_directory}/")
    
    # Inference loop
    total_time = 0.0
    for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
        # Load images
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        
        original_size = (image1.shape[2], image1.shape[3])  # (H, W)
        
        # Preprocess
        if model_height is not None and not args.resize_to_model:
            # Fixed size model without resize, use padding
            padder = InputPadder(image1.shape, divis_by=32)
            image1_padded, image2_padded = padder.pad(image1, image2)
            scale_factor = (1.0, 1.0)
            use_padder = True
        elif model_height is not None and args.resize_to_model:
            # Resize to model size
            image1_padded, scale_factor = preprocess_image(image1, model_height, model_width)
            image2_padded, _ = preprocess_image(image2, model_height, model_width)
            use_padder = False
        else:
            # Dynamic size model, use padding
            padder = InputPadder(image1.shape, divis_by=32)
            image1_padded, image2_padded = padder.pad(image1, image2)
            scale_factor = (1.0, 1.0)
            use_padder = True
        
        # Inference
        start_time = time.time()
        outputs = session.run(
            output_names,
            {
                input_names[0]: image1_padded.astype(np.float32),
                input_names[1]: image2_padded.astype(np.float32)
            }
        )
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        if args.verbose:
            print(f"\n{os.path.basename(imfile1)} - Inference time: {inference_time:.4f} seconds")
        
        # Get disparity map
        disp = outputs[0]  # [B, 1, H, W]
        
        # Post-process
        if use_padder:
            disp = padder.unpad(disp)
        
        disp = postprocess_disparity(disp, original_size, scale_factor)
        
        # Save results
        file_stem = Path(imfile1).stem
        
        if args.save_png:
            # Save as 16-bit PNG (KITTI format)
            disp_png = np.round(disp * 256).astype(np.uint16)
            png_path = output_directory / f"{file_stem}.png"
            skimage.io.imsave(str(png_path), disp_png)
        
        if args.save_numpy:
            # Save as numpy array
            npy_path = output_directory / f"{file_stem}.npy"
            np.save(str(npy_path), disp.astype(np.float32))
        
        if args.save_colormap:
            # Save color visualization
            disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6) * 255
            disp_vis = disp_vis.astype(np.uint8)
            disp_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
            colormap_path = output_directory / f"{file_stem}_color.png"
            cv2.imwrite(str(colormap_path), disp_colormap)
    
    # Statistics
    avg_time = total_time / len(left_images)
    print(f"\nInference completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average per frame: {avg_time:.4f} seconds ({1.0/avg_time:.2f} FPS)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-MonSter++ ONNX Inference Script')
    
    # ONNX model path
    parser.add_argument('--onnx_path', type=str, default='./rt-monster_544_960.onnx',
                        help="ONNX model path")
    
    # Input images
    parser.add_argument('-l', '--left_imgs', type=str, required=True,
                        help="Left image path (supports wildcards)")
    parser.add_argument('-r', '--right_imgs', type=str, required=True,
                        help="Right image path (supports wildcards)")
    
    # Output settings
    parser.add_argument('--output_directory', type=str, default='./Output/onnx_output',
                        help="Output directory")
    parser.add_argument('--save_png', action='store_true', default=True,
                        help="Save as 16-bit PNG (enabled by default)")
    parser.add_argument('--save_numpy', action='store_true',
                        help="Save as numpy array")
    parser.add_argument('--save_colormap', action='store_true',
                        help="Save color visualization")
    
    # Inference settings
    parser.add_argument('--cpu', action='store_true',
                        help="Use CPU inference")
    parser.add_argument('--resize_to_model', action='store_true',
                        help="Resize image to model input size (otherwise use padding)")
    parser.add_argument('--verbose', action='store_true',
                        help="Show detailed information")
    
    args = parser.parse_args()
    
    demo(args)
