# 🚀 MonSter++ 🚀 
MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors  <a href="https://arxiv.org/abs/2501.08643"><img src="https://img.shields.io/badge/arXiv-2501.08643-b31b1b?logo=arxiv" alt='arxiv'></a>

<p align="center">
</p>
<h2>🤗 Demo Video：</h2>
<div align="center">
	<a href="https://www.youtube.com/watch?v=YIhQCuQAKrU">
		<img src="Media/demo.png" width="50%" alt="Overview Video">
	</a>
</div>
<p></p>

<div align="center">

</div>

# 🚀 RT-MonSter++ 🚀 
- `[2025/12]` We release the ONNX and TensorRT export and conversion scripts for RT-MonSter++, enabling efficient inference on mobile and edge devices. Further optimizations are planned, including integrating techniques from DepthAnything to further improve efficiency!
- <img width="300" height="220" alt="45a82bb73e85fb4a66ade8bd3c441b48" src="https://github.com/user-attachments/assets/c34a5e24-8327-4b61-9e17-66d286e2be2c" />

## News 
- `[2025/12]` The most often asked about the issue: ONNX and tensorRT export problems: can be reference to https://github.com/Junda24/MonSter-plusplus/pull/4
- `[2025/9]` We have open-sourced our lightweight real-time model RT-MonSter++
- `[2025/9]` Weights for RT-MonSter++ model released!
- `[2025/10]` We have open-sourced our accuracy model MonSter++
- `[2025/10]` Weights for MonSter++ model released!

## ✈️ MonSter++ Model weights (accuracy models)

| Model      |                                               Link                                                |
|:----:|:-------------------------------------------------------------------------------------------------:|
| KITTI| [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_large.pth?download=true) |
| ETH3D | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/ETH3D_large.pth?download=true)|
|Middlebury | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Middlebury_large.pth?download=true)|
|mix_all | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Mix_all_large.pth?download=true)|

The mix_all model is trained on all the datasets we collect over 2M image pairs, which has the best performance on zero-shot generalization.

## ✈️ RT-MonSter++ Model weights (light weight models)

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

1. [KITTI 2012 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
<img width="1221" height="696" alt="image" src="https://github.com/user-attachments/assets/886445d2-c9c2-4148-9bd2-599e62802e96" />

2. [KITTI 2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

<img width="1205" height="709" alt="image" src="https://github.com/user-attachments/assets/161d5344-0a10-4e93-9aa4-ea99f9bfb349" />

 We obtained the 2nd place on the world-wide [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view), while maintaining the lowest inference cost, particularly compared with the top-ranked method.

 3. [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view)
 <img width="2543" height="1363" alt="3e06f3c5a624ab19c78fb89c0f516ed2" src="https://github.com/user-attachments/assets/8168ee8c-6003-4ca8-8efe-b85030139017" />


## ✈️ Citation

If you find our works useful in your research, please consider citing our papers:

```bibtex

MonSter：
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Junda and Liu, Longliang and Xu, Gangwei and Wang, Xianqi and Zhang, Zhaoxing and Deng, Yong and Zang, Jinliang and Chen, Yurui and Cai, Zhipeng and Yang, Xin},
    title     = {MonSter: Marry Monodepth to Stereo Unleashes Power},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6273-6282}
}

MonSter++：
@misc{cheng2025monsterunifiedstereomatching,
      title={MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors}, 
      author={Junda Cheng and Wenjing Liao and Zhipeng Cai and Longliang Liu and Gangwei Xu and Xianqi Wang and Yuzhou Wang and Zikang Yuan and Yong Deng and Jinliang Zang and Yangyang Shi and Jinhui Tang and Xin Yang},
      year={2025},
      eprint={2501.08643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.08643}, 
}
```
