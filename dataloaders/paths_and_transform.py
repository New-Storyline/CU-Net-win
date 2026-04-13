# ------------------------------------------------------------
# dataloader for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------

import os
import glob
import torch
from dataloaders import transforms


GEOMETRIC = ['BottomCrop', 'HorizontalFlip']
TRAIN_D_AND_GT = ['GEOMETRIC', 'Random_crop']
TRIAN_TRANSFORM_RGB = ['ColorJitter', 'GEOMETRIC', 'Random_crop']
TRIAN_TRANSFORM_STRUCTURE = ['ColorJitter', 'GEOMETRIC', 'Random_crop']
TRAIN_POSITION = ['BottomCrop', 'Random_crop']

VAL_TRANSFORME = ['BottomCrop']

NO_TRANSFORM = []

DEBUG_DATALOADER = False
glob_d, glob_gt, glob_rgb, glob_s2r, transform = None, None, None, None, None
glob_penettruth, glob_s2dtruth = None, None
get_rgb_paths = None
pnew = None
get_penettruth_paths, get_s2dtruth_paths = None, None


def get_paths_and_transform(split, args):
    global glob_d, glob_gt, glob_rgb, glob_s2r, transform, get_rgb_paths, \
        get_penettruth_paths, get_s2dtruth_paths, glob_penettruth, glob_s2dtruth

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )
        glob_s2r = os.path.join(
            args.data_folder,
            'train/*_sync/image_0[2,3]/structure_data/*.npy'
        )
        def get_rgb_paths(p):
            global pnew
            if 'image_02' in p:
                pnew = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
            elif 'image_03' in p:
                pnew = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
            return pnew
        def get_penettruth_paths(p):
            pnew_ = p.replace('groundtruth', 'penettruth')
            return pnew_
        def get_s2dtruth_paths(p):
            pnew_ = p.replace('groundtruth', 's2dtruth')
            return pnew_
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            glob_s2r = os.path.join(
                args.data_folder,
                'val/*_sync/image_0[2,3]/structure_data/*.npy'
            )
            def get_rgb_paths(p):
                global pnew
                if 'image_02' in p:
                    pnew = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
                elif 'image_03' in p:
                    pnew = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
                return pnew
            def get_penettruth_paths(p):
                pnew_ = p.replace('groundtruth', 'penettruth')
                return pnew_
            def get_s2dtruth_paths(p):
                pnew_ = p.replace('groundtruth', 's2dtruth')
                return pnew_
        elif args.val == "select":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/velodyne_raw/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/groundtruth_depth/*.png'
            )
            glob_rgb = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/image/*.png'
            )
            glob_s2r = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/image_structure/*.npy'
            )
            glob_penettruth = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/penettruth_depth/*.png'
            )
            glob_s2dtruth = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/s2dtruth_depth/*.png'
            )
    elif split == "test_completion":
        transform = test_transform
        glob_d = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png'
        )
        glob_gt = None
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/image/*.png'
        )
        glob_s2r = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/image_structure/*.npy'
        )
    elif split == "test_prediction":
        transform = no_transform
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_prediction_anonymous/image/*.png'
        )
        glob_d = None
        glob_gt = None
        glob_s2r = None
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        if split == 'train' or (split == 'val' and args.val == 'full'):
            paths_rgb = [get_rgb_paths(p) for p in paths_d]
        else:
            paths_rgb = sorted(glob.glob(glob_rgb))
        if split == 'train' or (split == 'val' and args.val == 'full'):
            paths_penettruth = [get_penettruth_paths(p) for p in paths_gt]
            paths_s2dtruth = [get_s2dtruth_paths(p) for p in paths_gt]   
        else:
            paths_penettruth = sorted(glob.glob(glob_penettruth))
            paths_s2dtruth = sorted(glob.glob(glob_s2dtruth))      
        paths_s2r = sorted(glob.glob(glob_s2r))
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_s2r = sorted(glob.glob(glob_s2r))
        paths_gt = [None] * len(paths_rgb)
        paths_penettruth = [None] * len(paths_rgb)
        paths_s2dtruth = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))
            
    # DEBUG
    if DEBUG_DATALOADER:
        for i in range(999):
            print("#####")
            print(paths_rgb[i])
            print(paths_d[i])
            print(paths_gt[i])
            print(paths_s2r[i])
        raise OSError('debug end!')

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_s2r) == 0\
            and len(paths_penettruth) == 0 and len(paths_s2dtruth) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_s2r) == 0:
        raise (RuntimeError("Requested structure images but no structure was found"))
    if len(paths_penettruth) == 0:
        raise (RuntimeError("Requested penettruth images but no penettruth was found"))
    if len(paths_s2dtruth) == 0:
        raise (RuntimeError("Requested s2dtruth images but no s2dtruth was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt) or len(paths_gt) != len(paths_s2r)\
            or len(paths_gt) != len(paths_penettruth) or len(paths_gt) != len(paths_s2dtruth):
        print(len(paths_d), len(paths_gt), len(paths_penettruth), len(paths_s2dtruth), len(paths_rgb), len(paths_s2r))

    paths = {"rgb": paths_rgb, "dep": paths_d, "gt": paths_gt, 'penetgt': paths_penettruth, 's2dgt': paths_s2dtruth,
             "structure": paths_s2r}

    items = {}
    for key, val in paths.items():
        if key not in args.dataset:
            items[key] = [None] * len(paths_rgb)
        else:
            items[key] = val

    return items, transform

def _crop_array(arr, h, rheight, j, rwidth):
    if arr is None:
        return None
    if arr.ndim == 3:
        return arr[h - rheight:h, j:j + rwidth, :]
    elif arr.ndim == 2:
        return arr[h - rheight:h, j:j + rwidth]
    return arr


def _try_apply(arr, transform_fn):
    if arr is not None:
        return transform_fn(arr)
    return None


def _apply_transform_all(transform_fn, sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position):
    return (
        _try_apply(sparse, transform_fn),
        _try_apply(gt, transform_fn),
        _try_apply(ipbasicgt, transform_fn),
        _try_apply(penetgt, transform_fn),
        _try_apply(s2dgt, transform_fn),
        _try_apply(rgb, transform_fn),
        _try_apply(structure, transform_fn),
        _try_apply(position, transform_fn),
    )


def _random_crop_all(h, w, rheight, rwidth, sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position):
    j = int(torch.FloatTensor(1).uniform_(0, w - rwidth + 1).item())
    return (
        _crop_array(sparse, h, rheight, j, rwidth),
        _crop_array(gt, h, rheight, j, rwidth),
        _crop_array(ipbasicgt, h, rheight, j, rwidth),
        _crop_array(penetgt, h, rheight, j, rwidth),
        _crop_array(s2dgt, h, rheight, j, rwidth),
        _crop_array(rgb, h, rheight, j, rwidth),
        _crop_array(structure, h, rheight, j, rwidth),
        _crop_array(position, h, rheight, j, rwidth),
    )


def _make_jitter_transform(args, geometric_transform):
    brightness = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter), 1 + args.jitter).item()
    contrast = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter), 1 + args.jitter).item()
    saturation = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter), 1 + args.jitter).item()
    return transforms.Compose([
        transforms.ColorJitter(brightness, contrast, saturation, 0),
        geometric_transform
    ])


def train_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = args.val_h
    owidth = args.val_w

    # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    flip = torch.FloatTensor(1).uniform_(0, 1)
    do_flip = flip.item() < 0.5

    # flip = np.random.uniform(0.0, 1.0)
    # do_flip = flip < 0.5
    # ic(flip)

    transforms_list = [
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]

    # if small_training == True:
    # transforms_list.append(transforms.RandomCrop((rheight, rwidth)))

    transform_geometric = transforms.Compose(transforms_list)

    sparse = _try_apply(sparse, transform_geometric)
    gt = _try_apply(gt, transform_geometric)
    ipbasicgt = _try_apply(ipbasicgt, transform_geometric)
    penetgt = _try_apply(penetgt, transform_geometric)
    s2dgt = _try_apply(s2dgt, transform_geometric)
    rgb = _try_apply(rgb, _make_jitter_transform(args, transform_geometric))
    structure = _try_apply(structure, _make_jitter_transform(args, transform_geometric))
    # sparse = drop_depth_measurements(sparse, 0.9)

    position = _try_apply(position,
        transforms.Compose([transforms.BottomCrop((oheight, owidth))]))

    # random crop
    if not args.not_random_crop:
        sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position = \
            _random_crop_all(oheight, owidth, args.random_crop_height, args.random_crop_width,
                             sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position)

    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

def _crop_and_optional_random(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args, skip_random_crop):
    oheight = args.val_h
    owidth = args.val_w

    transform_ = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
    sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position = \
        _apply_transform_all(transform_, sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position)

    if not skip_random_crop:
        sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position = \
            _random_crop_all(oheight, owidth, args.random_crop_height, args.random_crop_width,
                             sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position)

    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position


def val_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    return _crop_and_optional_random(
        sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args, args.val_not_random_crop)


def test_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    return _crop_and_optional_random(
        sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args, args.test_not_random_crop)

def no_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()
