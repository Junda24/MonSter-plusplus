# MonSter++
MonSter++: A Unified Geometric Foundation Model for Stereo and Multi-View Depth Estimation via the Unleashing of Monodepth Priors

Code coming soon!
## News 
- `[2025/9]` We have open-sourced our lightweight real-time model RT-MonSter++
- `[2025/9]` Weights for RT-MonSter++ model released! ！

## ✈️ RT-MonSter++ Model weights (light weight model)

| Model      |                                               Link                                                |
|:----:|:-------------------------------------------------------------------------------------------------:|
| KITTI 2012| [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_2012.pth?download=true) |
| KITTI 2015 | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_2015.pth?download=true)|
|mix_all | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Zero_shot.pth?download=true)|

The mix_all model is trained on all the datasets we collect over 2M image pairs, which has the best performance on zero-shot generalization.

### 🎬 Dependencies

```Shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.6.13
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install openexr
pip install pyexr
pip install imath
pip install h5py
pip install swanlab

```

# Leaderboards 🏆
We obtained the 1st place on the world-wide [KITTI 2012 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

[KITTI 2012 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
<img width="1221" height="696" alt="image" src="https://github.com/user-attachments/assets/886445d2-c9c2-4148-9bd2-599e62802e96" />

[KITTI 2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

<img width="1205" height="709" alt="image" src="https://github.com/user-attachments/assets/161d5344-0a10-4e93-9aa4-ea99f9bfb349" />
 


