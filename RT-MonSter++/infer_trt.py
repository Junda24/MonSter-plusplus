"""
RT-MonSter++ TensorRT Inference Script

Usage:
    # Single image pair inference
    python infer_trt.py --trt_path ./rt-monster_544_960.trt \
        -l ./demo_img/left.png -r ./demo_img/right.png \
        --output_directory ./Output/trt_infer_demo \
        --save_colormap \
        --save_numpy

    # Batch image inference
    python infer_trt.py --trt_path ./rt-monster_544_960.trt \
        -l "/data/StereoData/XXX/left/*.png" \
        -r "/data/StereoData/XXX/right/*.png" \
        --output_directory ./Output/trt_infer_results
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
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"Please install TensorRT and PyCUDA: {e}")
    print("Installation:")
    print("  pip install nvidia-tensorrt pycuda")
    print("  or refer to NVIDIA official documentation to install TensorRT")
    sys.exit(1)


# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
    """TensorRT Inference Wrapper Class"""
    
    def __init__(self, trt_path):
        """
        Initialize TensorRT inference engine
        
        Args:
            trt_path: TensorRT engine file path (.trt)
        """
        self.trt_path = trt_path
        self.engine = None
        self.context = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine"""
        print(f"Loading TensorRT engine: {self.trt_path}")
        
        with open(self.trt_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.trt_path}")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        print(f"TensorRT engine loaded successfully")
    
    def _allocate_buffers(self):
        """Allocate input/output buffers"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem
                })
                print(f"Input: {name}, shape: {shape}, dtype: {dtype}")
            else:
                self.outputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem
                })
                print(f"Output: {name}, shape: {shape}, dtype: {dtype}")
    
    def get_input_shape(self):
        """Get model input size"""
        if len(self.inputs) > 0:
            shape = self.inputs[0]['shape']
            return shape[2], shape[3]  # height, width
        return None, None
    
    def infer(self, left_image, right_image):
        """
        Execute inference
        
        Args:
            left_image: numpy array [B, C, H, W]
            right_image: numpy array [B, C, H, W]
        
        Returns:
            disparity: numpy array [B, 1, H, W]
        """
        # Copy input data to host memory
        np.copyto(self.inputs[0]['host'], left_image.ravel())
        np.copyto(self.inputs[1]['host'], right_image.ravel())
        
        # Transfer input data to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer output data to CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Get output
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return output.copy()
    
    def __del__(self):
        """Release resources"""
        # PyCUDA automatically manages memory release
        pass


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


def preprocess_image(image, target_height, target_width):
    """
    Preprocess image to fit model input
    
    Args:
        image: numpy array [B, C, H, W]
        target_height: target height
        target_width: target width
    
    Returns:
        processed_image: numpy array [B, C, H, W]
        scale_factor: (scale_h, scale_w) for post-processing
    """
    _, _, h, w = image.shape
    scale_h, scale_w = 1.0, 1.0
    
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
    
    # Check if TensorRT engine exists
    if not os.path.exists(args.trt_path):
        print(f"Error: TensorRT engine file not found: {args.trt_path}")
        sys.exit(1)
    
    # Create TensorRT inference engine
    trt_infer = TRTInference(args.trt_path)
    
    # Get expected model input size
    model_height, model_width = trt_infer.get_input_shape()
    if model_height is not None:
        print(f"Model input size: {model_height} x {model_width}")
    
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
    
    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} iterations)...")
        dummy_left = np.random.randn(1, 3, model_height, model_width).astype(np.float32)
        dummy_right = np.random.randn(1, 3, model_height, model_width).astype(np.float32)
        for _ in range(args.warmup):
            _ = trt_infer.infer(dummy_left, dummy_right)
        print("Warmup completed")
    
    # Inference loop
    total_time = 0.0
    for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
        # Load images
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        
        original_size = (image1.shape[2], image1.shape[3])  # (H, W)
        
        # Preprocess - TensorRT requires fixed size input, must resize
        image1_resized, scale_factor = preprocess_image(image1, model_height, model_width)
        image2_resized, _ = preprocess_image(image2, model_height, model_width)
        
        # Inference
        cuda.Context.synchronize()
        start_time = time.time()
        disp = trt_infer.infer(
            image1_resized.astype(np.float32),
            image2_resized.astype(np.float32)
        )
        cuda.Context.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        if args.verbose:
            print(f"\n{os.path.basename(imfile1)} - Inference time: {inference_time:.4f} seconds")
        
        # Post-process
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
    parser = argparse.ArgumentParser(description='RT-MonSter++ TensorRT Inference Script')
    
    # TensorRT engine path
    parser.add_argument('--trt_path', type=str, default='./rt-monster_544_960.trt',
                        help="TensorRT engine path (.trt)")
    
    # Input images
    parser.add_argument('-l', '--left_imgs', type=str, required=True,
                        help="Left image path (supports wildcards)")
    parser.add_argument('-r', '--right_imgs', type=str, required=True,
                        help="Right image path (supports wildcards)")
    
    # Output settings
    parser.add_argument('--output_directory', type=str, default='./Output/trt_output',
                        help="Output directory")
    parser.add_argument('--save_png', action='store_true', default=True,
                        help="Save as 16-bit PNG (enabled by default)")
    parser.add_argument('--save_numpy', action='store_true',
                        help="Save as numpy array")
    parser.add_argument('--save_colormap', action='store_true',
                        help="Save color visualization")
    
    # Inference settings
    parser.add_argument('--warmup', type=int, default=3,
                        help="Warmup iterations (default 3)")
    parser.add_argument('--verbose', action='store_true',
                        help="Show detailed information")
    
    args = parser.parse_args()
    
    demo(args)
