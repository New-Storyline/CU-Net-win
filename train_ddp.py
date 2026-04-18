# CONFIG (parsed before torch import so CUDA_VISIBLE_DEVICES takes effect)
import argparse
import os

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='CUNet')
arg.add_argument('-n', '--model_name', type=str, default='CUNet')
arg.add_argument('-c', '--configuration', type=str, default='train.yml')
arg = arg.parse_args()

from configs import get as get_cfg
config = get_cfg(arg)
assert len(config.output) != 0, 'the output of network should not be zero!'
if config.output == 'ben_depth':
    assert len(config.ben_supervised) != 0 and len(config.jin_supervised) == 0 and len(config.an_supervised) == 0
elif config.output == 'jin_depth':
    assert len(config.jin_supervised) != 0 and len(config.an_supervised) == 0
elif config.output == 'an_depth':
    assert len(config.an_supervised) != 0

# ENVIRONMENT (must be set before torch import)
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(g) for g in config.gpus)
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = str(config.port)
if not config.record_by_wandb_online:
    os.environ["WANDB_MODE"] = 'dryrun'

import time
import random
import numpy as np
import emoji
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloaders.custom_dataset import CustomDepthDataset
from dataloaders.utils import load_calib
from model import get as get_model
from optimizer_scheduler import make_optimizer_scheduler
from summary import get as get_summary
from metric import get as get_metric
from utility import *
from loss import get as get_loss


# ---------------------------------------------------------------------------
#  Setup helpers
# ---------------------------------------------------------------------------

def setup_distributed(gpu, args):
    torch.cuda.set_device(gpu)
    if not args.no_multiprocessing:
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        dist.init_process_group(backend=backend, init_method='env://',
                                world_size=args.num_gpus, rank=gpu)


def set_seed(args):
    rank = 0 if args.no_multiprocessing else dist.get_rank()
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


def setup_wandb(args):
    import wandb
    wandb.login()
    if args.resume:
        if len(args.wandb_id_resume) == 0:
            wandb.init(dir=ROOT_PATH, config=args,
                       project=args.project_name, resume=True)
        else:
            wandb.init(id=args.wandb_id_resume, dir=ROOT_PATH, config=args,
                       project=args.project_name, resume='must')
        print(f'=> Resume wandb from : {args.wandb_id_resume}')
    else:
        wandb.init(dir=ROOT_PATH, config=args, project=args.project_name)

    args.defrost()
    args.save_dir = os.path.split(wandb.run.dir)[0]
    args.freeze()

    if not args.resume:
        with open(os.path.join(args.save_dir, 'config.txt'), 'w') as f:
            f.write(args.dump())


# ---------------------------------------------------------------------------
#  Factory helpers
# ---------------------------------------------------------------------------

def load_paths_from_csv(csv_path, base_dir):
    """Read a CSV file with rows ``rgb_path,depth_path`` and return a paths dict."""
    import csv
    rgb_paths, depth_paths = [], []
    with open(csv_path, 'r') as f:
        for row in csv.reader(f):
            rgb_paths.append(os.path.join(base_dir, row[0]))
            depth_paths.append(os.path.join(base_dir, row[1]))
    return {'rgb': rgb_paths, 'gt_depth': depth_paths}


def make_sparse_depth(depth_map, keep_ratio=0.05):
    """Randomly zero-out depth pixels to simulate sparse measurements."""
    mask = np.random.uniform(size=depth_map.shape) < keep_ratio
    sparse = depth_map.copy()
    sparse[~mask] = 0.0
    return sparse


def make_simple_transform(image_size):
    """Return a transform that center-crops / resizes all arrays to *image_size*."""
    from dataloaders import transforms
    crop = transforms.Compose([transforms.BottomCrop(image_size)])

    def transform_fn(sparse, gt, rgb, position):
        if sparse is not None:
            sparse = crop(sparse)
        if gt is not None:
            gt = crop(gt)
        if rgb is not None:
            rgb = crop(rgb)
        if position is not None:
            position = crop(position)
        return sparse, gt, rgb, position

    return transform_fn


def create_custom_dataset(csv_path, base_dir, args):
    """Build a ``CustomDepthDataset`` from a CSV listing."""
    image_size = (args.val_h, args.val_w)
    return CustomDepthDataset(
        image_size=image_size,
        get_image_pathes_fn=lambda: load_paths_from_csv(csv_path, base_dir),
        transform_fn=make_simple_transform(image_size),
        create_sparse_depth_fn=make_sparse_depth,
        load_calib_fn=load_calib,
    )


def create_dataloaders(args):
    batch_size = args.batch_size if args.no_multiprocessing else args.batch_size // args.num_gpus

    data_train = create_custom_dataset(
        csv_path=os.path.join(args.data_folder, 'nyu2_train.csv'),
        base_dir=args.data_folder,
        args=args,
    )
    if args.no_multiprocessing:
        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_threads, pin_memory=True, drop_last=True)
    else:
        sampler_train = DistributedSampler(data_train)
        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False,
                                  num_workers=args.num_threads, pin_memory=True,
                                  sampler=sampler_train, drop_last=True)

    data_val = create_custom_dataset(
        csv_path=os.path.join(args.data_folder, 'nyu2_test.csv'),
        base_dir=args.data_folder,
        args=args,
    )
    if args.no_multiprocessing:
        loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False,
                                num_workers=args.num_threads, pin_memory=True)
    else:
        sampler_val = SequentialDistributedSampler(data_val, batch_size=batch_size)
        loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False,
                                num_workers=args.num_threads, pin_memory=True,
                                sampler=sampler_val)

    return loader_train, loader_val, batch_size


def create_model(gpu, args):
    model_cls = get_model(args)
    net = model_cls(args)
    checkpoint = None
    if gpu == 0:
        count_parameters(net)
        if len(args.pretrain) != 0:
            assert os.path.isfile(args.pretrain), f"file not found: {args.pretrain}"
            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])
            print(f'=> Load network parameters from : {args.pretrain}')
    if args.syncbn and not args.no_multiprocessing:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda(gpu)
    if not args.no_multiprocessing:
        net = DDP(net, device_ids=[gpu], output_device=gpu)
    return net, checkpoint


def create_losses(args):
    loss_cls = get_loss(args)
    loss_ben = loss_cls(args, args.loss_ben)
    loss_ben.cuda()

    loss_jin = None
    if len(args.jin_supervised) != 0:
        loss_jin = loss_cls(args, args.loss_jin)
        loss_jin.cuda()

    loss_an = None
    if len(args.an_supervised) != 0:
        loss_an = loss_cls(args, args.loss_an)
        loss_an.cuda()

    return loss_ben, loss_jin, loss_an


def create_summary_writers(args, loss_ben, loss_jin, loss_an, metric):
    summary = get_summary(args)
    writers = {
        'ben_train': summary(args.save_dir, 'ben_train', args, loss_ben.loss_name, metric.metric_name),
        'ben_val':   summary(args.save_dir, 'ben_val',   args, loss_ben.loss_name, metric.metric_name),
    }
    if args.summary_jin:
        writers['jin_train'] = summary(args.save_dir, 'jin_train', args, loss_jin.loss_name, metric.metric_name)
        writers['jin_val']   = summary(args.save_dir, 'jin_val',   args, loss_jin.loss_name, metric.metric_name)
    if args.summary_an:
        writers['an_train'] = summary(args.save_dir, 'an_train', args, loss_an.loss_name, metric.metric_name)
        writers['an_val']   = summary(args.save_dir, 'an_val',   args, loss_an.loss_name, metric.metric_name)
    return writers


# ---------------------------------------------------------------------------
#  Loss / checkpoint helpers
# ---------------------------------------------------------------------------

def compute_all_losses(args, output, sample, loss_ben, loss_jin, loss_an):
    loss_sum_ben, loss_val_ben = loss_ben(output['ben_depth'], sample[args.ben_supervised])
    zero = torch.tensor(0.0)

    if loss_jin is not None:
        loss_sum_jin, loss_val_jin = loss_jin(output['jin_depth'], sample[args.jin_supervised])
    else:
        loss_sum_jin, loss_val_jin = zero, zero

    if loss_an is not None:
        loss_sum_an, loss_val_an = loss_an(output['an_depth'], sample[args.an_supervised])
    else:
        loss_sum_an, loss_val_an = zero, zero

    return loss_sum_ben, loss_val_ben, loss_sum_jin, loss_val_jin, loss_sum_an, loss_val_an


def compute_weighted_loss(epoch, args, loss_sum_ben, loss_sum_jin, loss_sum_an):
    if epoch <= args.round1:
        wb, wj, wa = args.weight_ben1, args.weight_jin1, args.weight_an1
    elif epoch <= args.round2:
        wb, wj, wa = args.weight_ben2, args.weight_jin2, args.weight_an2
    else:
        wb, wj, wa = args.weight_ben3, args.weight_jin3, args.weight_an3
    return wb * loss_sum_ben + wj * loss_sum_jin + wa * loss_sum_an


def make_checkpoint(args, net, optimizer, scheduler, epoch, log_itr):
    return {
        'net': net.state_dict() if args.no_multiprocessing else net.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'log_itr': log_itr,
        'args': args,
    }


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def _gather_results(args, loss_list, metric_list, num_samples):
    losses = torch.cat(loss_list, dim=0)
    metrics = torch.cat(metric_list, dim=0)
    if args.no_multiprocessing:
        return losses, metrics
    return distributed_concat(losses, num_samples), distributed_concat(metrics, num_samples)


def should_validate(epoch, batch_idx, loader_train, args):
    is_last_batch = (batch_idx == len(loader_train) - 1)
    if epoch <= args.val_epoch:
        return is_last_batch
    num_gpus_mult = 1 if args.no_multiprocessing else args.num_gpus
    val_interval = args.val_iters // (loader_train.batch_size * num_gpus_mult)
    return (batch_idx + 1) % val_interval == 0


def run_validation(gpu, args, net, loader_val, batch_size,
                   loss_ben, loss_jin, loss_an, metric, writers, log_val):
    torch.set_grad_enabled(False)
    net.eval()
    num_gpus_mult = 1 if args.no_multiprocessing else args.num_gpus

    pbar_val = None
    if gpu == 0:
        pbar_val = tqdm(total=len(loader_val) * loader_val.batch_size * num_gpus_mult)

    loss_ben_list, metric_ben_list = [], []
    loss_jin_list, metric_jin_list = [], []
    loss_an_list, metric_an_list = [], []

    loss_sum_ben_val = torch.tensor(0.0)
    loss_sum_jin_val = torch.tensor(0.0)
    loss_sum_an_val = torch.tensor(0.0)
    sample_val = output_val = None

    for _, sample_val in enumerate(loader_val):
        sample_val = {k: v.cuda(gpu) for k, v in sample_val.items() if v is not None}
        output_val = net(sample_val)

        for ben_depth, ben_sup, gt in zip(
                torch.chunk(output_val['ben_depth'], batch_size, dim=0),
                torch.chunk(sample_val[args.ben_supervised], batch_size, dim=0),
                torch.chunk(sample_val['gt'], batch_size, dim=0)):
            loss_sum_ben_val, loss_val_ben_val = loss_ben(ben_depth, ben_sup)
            loss_ben_list.append(loss_val_ben_val)
            metric_ben_list.append(metric.evaluate(ben_depth, gt, 'val'))

        if args.summary_jin:
            for jin_depth, jin_sup, gt in zip(
                    torch.chunk(output_val['jin_depth'], batch_size, dim=0),
                    torch.chunk(sample_val[args.jin_supervised], batch_size, dim=0),
                    torch.chunk(sample_val['gt'], batch_size, dim=0)):
                loss_sum_jin_val, loss_val_jin_val = loss_jin(jin_depth, jin_sup)
                loss_jin_list.append(loss_val_jin_val)
                metric_jin_list.append(metric.evaluate(jin_depth, gt, 'val'))

        if args.summary_an:
            for an_depth, an_sup, gt in zip(
                    torch.chunk(output_val['an_depth'], batch_size, dim=0),
                    torch.chunk(sample_val[args.an_supervised], batch_size, dim=0),
                    torch.chunk(sample_val['gt'], batch_size, dim=0)):
                loss_sum_an_val, loss_val_an_val = loss_an(an_depth, an_sup)
                loss_an_list.append(loss_val_an_val)
                metric_an_list.append(metric.evaluate(an_depth, gt, 'val'))

        if gpu == 0:
            pbar_val.set_description(
                f'Val|Lb={loss_sum_ben_val.item():.4f}'
                f'|Lj={loss_sum_jin_val.item():.4f}'
                f'|La={loss_sum_an_val.item():.4f}')
            pbar_val.update(loader_val.batch_size * num_gpus_mult)

    # Gather & log
    num_val = len(loader_val.dataset)
    loss_ben_all, metric_ben_all = _gather_results(args, loss_ben_list, metric_ben_list, num_val)

    val_metric = None
    if gpu == 0:
        pbar_val.close()
        for l, m in zip(loss_ben_all, metric_ben_all):
            writers['ben_val'].add(l.unsqueeze(0), m.unsqueeze(0))
        val_metric = writers['ben_val'].update(
            log_val, sample_val, output_val,
            online_loss=args.ben_online_loss, online_metric=args.ben_online_metric,
            online_rmse_only=args.ben_online_rmse_only, online_img=args.ben_online_img)

    if args.summary_jin:
        loss_jin_all, metric_jin_all = _gather_results(args, loss_jin_list, metric_jin_list, num_val)
        if gpu == 0:
            for l, m in zip(loss_jin_all, metric_jin_all):
                writers['jin_val'].add(l.unsqueeze(0), m.unsqueeze(0))
            jin_val_metric = writers['jin_val'].update(
                log_val, sample_val, output_val,
                online_loss=args.jin_online_loss, online_metric=args.jin_online_metric,
                online_rmse_only=args.jin_online_rmse_only, online_img=args.jin_online_img)
            if args.output == 'jin_depth':
                val_metric = jin_val_metric

    if args.summary_an:
        loss_an_all, metric_an_all = _gather_results(args, loss_an_list, metric_an_list, num_val)
        if gpu == 0:
            for l, m in zip(loss_an_all, metric_an_all):
                writers['an_val'].add(l.unsqueeze(0), m.unsqueeze(0))
            an_val_metric = writers['an_val'].update(
                log_val, sample_val, output_val,
                online_loss=args.an_online_loss, online_metric=args.an_online_metric,
                online_rmse_only=args.an_online_rmse_only, online_img=args.an_online_img)
            if args.output == 'an_depth':
                val_metric = an_val_metric

    torch.set_grad_enabled(True)
    net.train()
    return val_metric, sample_val, output_val


# ---------------------------------------------------------------------------
#  Main training function (one per GPU)
# ---------------------------------------------------------------------------

def train(gpu, args):
    if gpu == 0:
        print(args.dump())

    setup_distributed(gpu, args)
    set_seed(args)

    # WANDB
    if gpu == 0:
        setup_wandb(args)

    # DATA
    if gpu == 0:
        print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"))
    loader_train, loader_val, batch_size = create_dataloaders(args)
    num_gpus_mult = 1 if args.no_multiprocessing else args.num_gpus
    if gpu == 0:
        print(f'=> Train dataset: {len(loader_train) * batch_size * num_gpus_mult} samples')
        print(f'=> Val dataset: {len(loader_val) * batch_size * num_gpus_mult} samples')

    # MODEL
    if gpu == 0:
        print(emoji.emojize('Prepare network... :writing_hand:', variant="emoji_type"))
    net, checkpoint = create_model(gpu, args)

    # OPTIMIZER
    if gpu == 0:
        print(emoji.emojize('Prepare optimizer... :writing_hand:', variant="emoji_type"))
    optimizer, scheduler = make_optimizer_scheduler(args, net)

    # RESUME
    if gpu == 0 and checkpoint is not None and args.resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.defrost()
            args.start_epoch = checkpoint['epoch'] + 1
            args.log_itr = checkpoint['log_itr']
            args.freeze()
            print(f'=> Resume optimizer, scheduler from : {args.pretrain}')
        except KeyError:
            print('=> State dicts for resume are not saved. Use --save_full argument')
        del checkpoint

    # LOSSES
    if gpu == 0:
        print(emoji.emojize('Prepare loss... :writing_hand:', variant="emoji_type"))
        print(f'=> Loss_ben: {args.loss_ben}; Loss_jin: {args.loss_jin}; Loss_an: {args.loss_an}')
    loss_ben, loss_jin, loss_an = create_losses(args)

    # METRIC
    if gpu == 0:
        print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"))
    metric_cls = get_metric(args)
    metric = metric_cls(args)

    # SUMMARY & OUTPUT DIRS
    writers = {}
    log_itr = args.log_itr
    best_metric = 1e8
    log_val = 0
    if gpu == 0:
        print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"))
        writers = create_summary_writers(args, loss_ben, loss_jin, loss_an, metric)
        backup_source_code(os.path.join(args.save_dir, 'backup_code'))
        os.makedirs(os.path.join(args.save_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'val'), exist_ok=True)
        print('=> Save backup source code and makedirs done')

    # WARM UP
    warm_up_cnt = 0.0
    warm_up_max_cnt = len(loader_train) + 1.0

    # TRAINING LOOP
    for epoch in range(args.start_epoch, args.epochs + 1):
        net.train()
        if not args.no_multiprocessing:
            loader_train.sampler.set_epoch(epoch)

        if gpu == 0:
            print(emoji.emojize("Let's do something interesting :oncoming_fist:", variant="emoji_type"))
            list_lr = [g['lr'] for g in optimizer.param_groups]
            current_time = time.strftime('%y%m%d@%H:%M:%S')
            print(f'=======> Epoch {epoch:5d} / {args.epochs:5d} | Lr : {list_lr} | {current_time} | {args.save_dir} <=======')
            pbar = tqdm(total=len(loader_train) * loader_train.batch_size * num_gpus_mult)

        for batch_idx, sample in enumerate(loader_train):
            sample = {k: v.cuda(gpu) for k, v in sample.items() if torch.is_tensor(v)}

            # WARM UP LR
            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * warm_up_cnt / warm_up_max_cnt

            output = net(sample)

            # LOSS
            loss_sum_ben, loss_val_ben, loss_sum_jin, loss_val_jin, loss_sum_an, loss_val_an = \
                compute_all_losses(args, output, sample, loss_ben, loss_jin, loss_an)
            loss_sum = compute_weighted_loss(epoch, args, loss_sum_ben, loss_sum_jin, loss_sum_an)

            # BACKWARD
            if args.accumulation_gradient:
                (loss_sum / args.accumulation_steps).backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss_sum.backward()
                optimizer.step()
                optimizer.zero_grad()

            # TRAIN METRICS & PROGRESS
            if gpu == 0:
                writers['ben_train'].add(loss_val_ben, metric.evaluate(output['ben_depth'], sample['gt'], 'train'), log_itr)
                if args.summary_jin:
                    writers['jin_train'].add(loss_val_jin, metric.evaluate(output['jin_depth'], sample['gt'], 'train'), log_itr)
                if args.summary_an:
                    writers['an_train'].add(loss_val_an, metric.evaluate(output['an_depth'], sample['gt'], 'train'), log_itr)
                log_itr += 1

                error_str = (f'Train|Ls={loss_sum.item():.4f}|Lb={loss_sum_ben.item():.4f}'
                             f'|Lj={loss_sum_jin.item():.4f}|La={loss_sum_an.item():.4f}')
                if epoch == 1 and args.warm_up:
                    error_str += f' | Lr Warm Up : {[round(g["lr"], 6) for g in optimizer.param_groups]}'
                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * num_gpus_mult)

            # VALIDATION
            if should_validate(epoch, batch_idx, loader_train, args):
                val_metric, sample_val, output_val = run_validation(
                    gpu, args, net, loader_val, batch_size,
                    loss_ben, loss_jin, loss_an, metric, writers, log_val)

                if gpu == 0:
                    # Save val images on the last validation of the epoch
                    val_interval = args.val_iters // (loader_train.batch_size * num_gpus_mult)
                    is_last_val = ((epoch <= args.val_epoch and batch_idx == len(loader_train) - 1)
                                   or (epoch > args.val_epoch
                                       and batch_idx + 1 == (len(loader_train) // val_interval) * val_interval))
                    if is_last_val:
                        writers['ben_val'].save(epoch, batch_idx + 1, sample_val, output_val)

                    if val_metric < best_metric:
                        best_metric = val_metric
                        torch.save(make_checkpoint(args, net, optimizer, scheduler, epoch, log_itr),
                                   os.path.join(args.save_dir, 'best_model.pt'))
                    log_val += 1

        # END OF EPOCH
        if gpu == 0:
            pbar.close()
            writers['ben_train'].update(epoch, sample, output,
                                        online_loss=args.ben_online_loss, online_metric=args.ben_online_metric,
                                        online_rmse_only=args.ben_online_rmse_only, online_img=args.ben_online_img)
            if args.summary_jin:
                writers['jin_train'].update(epoch, sample, output,
                                            online_loss=args.jin_online_loss, online_metric=args.jin_online_metric,
                                            online_rmse_only=args.jin_online_rmse_only, online_img=args.jin_online_img)
            if args.summary_an:
                writers['an_train'].update(epoch, sample, output,
                                           online_loss=args.an_online_loss, online_metric=args.an_online_metric,
                                           online_rmse_only=args.an_online_rmse_only, online_img=args.an_online_img)
            torch.save(make_checkpoint(args, net, optimizer, scheduler, epoch, log_itr),
                       os.path.join(args.save_dir, 'latest_model.pt'))
        scheduler.step()


def main(args):
    if args.no_multiprocessing:
        train(0, args)
    else:
        assert args.num_gpus > 0
        spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,), join=False)
        while not spawn_context.join():
            pass
        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
            process.join()


if __name__ == '__main__':
    main(config)
