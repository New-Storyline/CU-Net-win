from dataclasses import dataclass
import enum

import torch.utils.data as data

from dataloaders.utils import *
from dataloaders.paths_and_transform import *
from dataloaders import CoordConv

class CustomDepthDataset(data.Dataset):

    def __init__(
            self, 
            image_size: tuple,
            get_image_pathes_fn,
            transform_fn,
            create_sparse_depth_fn,
            load_calib_fn=None,
            depth_read_fn=depth_read,
            rgb_read_fn=rgb_read,
        ):
        """
        Create a custom dataset for depth completion.

        Args:
            get_image_pathes_fn: Callable that returns a dictionary with image
                paths for the dataset, for example `{'rgb': [...], 'gt_depth': [...]}`.
            create_sparse_depth_fn: Callable that converts a dense depth map to
                a sparse depth map. Expected signature:
                `(depth_map: np.ndarray) -> np.ndarray`.
            transform_fn: Callable that applies dataset transforms to the loaded
                arrays. Expected signature:
                `(sparse_depth, gt_depth, rgb, position) -> (sparse_depth, gt_depth, rgb, position)`.
            load_calib_fn: Callable that loads the camera calibration matrix K.
            depth_read_fn: Callable that reads a depth map from a file path.
            rgb_read_fn: Callable that reads an RGB image from a file path.
        """
        self.depth_read = depth_read_fn
        self.rgb_read = rgb_read_fn
        self.image_size = image_size
        self.paths = get_image_pathes_fn()
        self.create_sparse_depth = create_sparse_depth_fn
        self.transform = transform_fn
        self.K = load_calib_fn() if load_calib_fn is not None else None

    def __getraw__(self, index):
        """
        Load the raw data for a given index.
        """
        assert 'dep' in self.paths and 'gt_depth' in self.paths, "The paths dictionary must contain 'dep' and 'gt_depth' keys."

        gt = self.depth_read(self.paths['gt_depth'][index])
        rgb = self.rgb_read(self.paths['rgb'][index])
        sparse = self.create_sparse_depth(gt)

        return sparse, gt, rgb,  self.paths['gt_depth'][index], self.paths['rgb'][index]

    def __getitem__(self, index):
        """
        Get the transformed data for a given index.
        
        Returns:
            A dictionary containing the transformed data:
            ```
            {
                'rgb': Tensor of shape (C, H, W),
                'dep': Tensor of shape (1, H, W),
                'gt': Tensor of shape (1, H, W),
                'gray': Tensor of shape (1, H, W),
                'position': Tensor of shape (2, H, W),
                'K': Camera intrinsic matrix,
                'd_path': Path to the ground truth depth map,
                'rgb_path': Path to the RGB image,
            }
            ```
        """

        sparse_dirty, gt, rgb_raw, gt_depth_path, rgb_path = self.__getraw__(index)
        position = CoordConv.AddCoordsNp(self.image_size[0], self.image_size[1])
        position = position.call()
        rgb = rgb_raw

        sparse_dirty, gt, rgb, position = self.transform(sparse_dirty, gt, rgb, position)
        rgb, gray = handle_gray(rgb)

        candidates = {"rgb": rgb,  "dep": sparse_dirty, "gt": gt,
                      "gray": gray, 'position': position, 'K': self.K}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        items['d_path'] = gt_depth_path
        items['rgb_path'] = rgb_path

        return items

    def __len__(self):
        return len(self.paths['gt_depth'])

