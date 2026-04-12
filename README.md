# CU-Net on windows: LiDAR Depth-only Completion with Coupled U-Net

This repository contains modifications to the original implementation of the CU-Net approach for use on Windows.
The following changes are planned:
- [ ] Completely remove training parallelization.
- [ ] Remove functions that only work on Linux.
- [ ] Update the configuration file to reflect these changes.

## Introduction
LiDAR depth-only completion is a challenging task to estimate a dense depth map only from sparse measurement points obtained by LiDAR. Even though the depth-only completion methods have been widely developed, there is still a significant performance gap with the RGB-guided methods that utilize extra color images. We find that existing depth-only methods can obtain satisfactory results in the areas where the measurement points are almost accurate and evenly distributed (denoted as normal areas), while the performance is limited in the areas where the foreground and background points are overlapped due to occlusion (denoted as overlap areas) and the areas where there are no available measurement points around (denoted as blank areas), since the methods have no reliable input information in these areas.
Building upon these observations, we propose an effective Coupled U-Net (CU-Net) architecture for depth-only completion.
Instead of directly using a large network for regression, we employ the local U-Net to estimate accurate values in the normal areas and provide the global U-Net with reliable initial values in the overlap and blank areas. The depth maps predicted by the two coupled U-Nets complement each other and can be fused by learned confidence maps to obtain the final completion results. In addition, we propose a confidence-based outlier removal module, which identifies the regions with outliers and removes outliers using simple judgment conditions. The proposed method boosts the final dense depth maps with fewer parameters and achieves state-of-the-art results on the KITTI benchmark. Moreover, it owns a powerful generalization ability under various depth densities, varying lighting, and weather conditions.

## Dependencies

Our released implementation depends on the following packages:

- torch
- torchvision
- emoji
- yacs
- wandb
- scipy
- scikit-image
- tensorboard
- prettytable

## How to use

... To be continued ...

## Publication
If you use our code and benchmark in your academic work, please cite the corresponding [paper](https://ieeexplore.ieee.org/document/9866514):

    @ARTICLE{wang2022ral,
      author={Wang, Yufei and Dai, Yuchao and Liu, Qi and Yang, Peng and Sun, Jiadai and Li, Bo},
      journal={IEEE Robotics and Automation Letters}, 
      title={CU-Net: LiDAR Depth-Only Completion With Coupled U-Net}, 
      year={2022},
      volume={7},
      number={4},
      pages={11476-11483},
      doi={10.1109/LRA.2022.3201193}}
