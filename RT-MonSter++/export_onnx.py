"""
RT-MonSter++ ONNX Export Script

Requires:
    pip install onnx onnxruntime

Export ONNX Examples:
    1. Export with static shape  (e.g. 736x1280):
    python export_onnx.py --restore_ckpt /path_to_checkpoints/RT-Monster_Zero_shot.pth --output ./rt-monster_736_1280.onnx

    2. Export with dynamic shape (e.g. 544x960 to 736x1280):
    python export_onnx.py --restore_ckpt /path_to_checkpoints/RT-Monster_Zero_shot.pth --dynamic --output ./rt-monster_dynamic.onnx

Convert onnx to TensorRT:
    1. Static shape
    trtexec --onnx=rt-monster_736_1280.onnx --saveEngine=rt-monster_736_1280.trt --fp16
    
    2. Dynamic shape
    trtexec --onnx=rt-monster_dynamic.onnx \
        --minShapes=left_image:1x3x544x960,right_image:1x3x544x960 \
        --optShapes=left_image:1x3x736x1280,right_image:1x3x736x1280 \
        --maxShapes=left_image:1x3x736x1280,right_image:1x3x736x1280 \
        --saveEngine=rt-monster_dynamic.trt \
        --fp16
"""


import sys
import os

sys.path.append('core')

import argparse
import logging
import torch
import torch.nn as nn
from core.monster import Monster, autocast


class MonsterWrapper(nn.Module):
    """Simple model wrapper"""
    def __init__(self, model, iters=4):
        super().__init__()
        self.model = model
        self.iters = iters
    
    def forward(self, image1, image2):
        """
        Args:
            image1: left image [B, 3, H, W]
            image2: right image [B, 3, H, W]
        Returns:
            disp: disparity map [B, 1, H, W]
        """
        return self.model(image1, image2, iters=self.iters, test_mode=True)


def export_onnx(args):
    """Export model to ONNX format"""

    logging.info("Initializing model...")

    # Create model
    model = Monster(args)

    # Load weights
    if args.restore_ckpt is not None:
        logging.info(f"Loading weights: {args.restore_ckpt}")
        assert os.path.exists(args.restore_ckpt), f"Weight file does not exist: {args.restore_ckpt}"
        
        checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # Handle DataParallel "module." prefix
        new_ckpt = {}
        for key in checkpoint:
            if key.startswith("module."):
                new_ckpt[key[7:]] = checkpoint[key]
            else:
                new_ckpt[key] = checkpoint[key]
        
        model.load_state_dict(new_ckpt, strict=False)
        logging.info("Weights loaded successfully")

    model.eval()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")

    # Wrap model
    wrapper = MonsterWrapper(model, iters=args.valid_iters)
    wrapper.eval()
    wrapper = wrapper.to(device)
    
    # input size must be multiple of 32
    height = (args.height // 32) * 32
    width = (args.width // 32) * 32
    logging.info(f"input size: {height} x {width}")

    # dummy inputs
    dummy_left = torch.randn(1, 3, height, width, device=device)
    dummy_right = torch.randn(1, 3, height, width, device=device)

    # verify model inference
    logging.info("Verifying model inference...")
    with torch.no_grad():
        _ = wrapper(dummy_left, dummy_right)
    logging.info("Model verification passed")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export ONNX
    logging.info(f"Exporting ONNX to: {args.output}")

    input_names = ['left_image', 'right_image']
    output_names = ['disparity']
    
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'left_image': {0: 'batch', 2: 'height', 3: 'width'},
            'right_image': {0: 'batch', 2: 'height', 3: 'width'},
            'disparity': {0: 'batch', 2: 'height', 3: 'width'}
        }
    
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_left, dummy_right),
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=args.verbose
        )

    logging.info(f"ONNX export successful: {args.output}")

    # Verify ONNX model
    if args.verify:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            logging.info("Verifying ONNX model...")

            # 检查模型
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            logging.info("ONNX model check passed")

            # ONNX Runtime 推理
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(args.output, providers=providers)
            
            ort_outputs = ort_session.run(None, {
                'left_image': dummy_left.cpu().numpy(),
                'right_image': dummy_right.cpu().numpy()
            })
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = wrapper(dummy_left, dummy_right)
            
            # check difference
            diff = np.abs(pytorch_output.cpu().numpy() - ort_outputs[0])
            logging.info(f"Max difference: {np.max(diff):.6f}, Average difference: {np.mean(diff):.6f}")

        except ImportError as e:
            logging.warning(f"Skipping verification (missing dependency): {e}")
        except Exception as e:
            logging.warning(f"Verification error: {e}")

    logging.info("Export completed!")


def main():
    parser = argparse.ArgumentParser(description='RT-MonSter++ ONNX Export')

    # model settings (default values from config/train_zeroshot.yaml)
    parser.add_argument('--restore_ckpt', type=str, required=True, help="path to model checkpoint")
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[32, 64, 96])
    parser.add_argument('--corr_radius', type=list, default=[2, 2, 4])
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=416)
    parser.add_argument('--valid_iters', type=int, default=4)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--corr_implementation', default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--slow_fast_gru', action='store_true')
    
    # export settings
    parser.add_argument('--output', type=str, default='./rt_monster.onnx', help="path to output file")
    parser.add_argument('--height', type=int, default=736, help="input height")
    parser.add_argument('--width', type=int, default=1280, help="input width")
    parser.add_argument('--opset', type=int, default=17, help="ONNX opset version")
    parser.add_argument('--dynamic', action='store_true', help="enable dynamic shape")
    parser.add_argument('--verify', action='store_true', help="verify model")
    parser.add_argument('--cpu', action='store_true', help="use CPU")
    parser.add_argument('--verbose', action='store_true', help="verbose output")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    logging.info("=" * 50)
    logging.info("RT-MonSter++ ONNX Export")
    logging.info(f"restore checkpoint: {args.restore_ckpt}")
    logging.info(f"output: {args.output}")
    logging.info(f"size: {args.height} x {args.width}")
    logging.info("=" * 50)
    
    export_onnx(args)


if __name__ == '__main__':
    main()
