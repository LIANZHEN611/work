import copy
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
from lib.datasets.kitti.concat_dataset import CustomConcatDataset
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.target_dataset_eval import TargetDatasetEval
from lib.datasets.kitti.target_dataset_yolo import TargetDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _resolve_target_root_dirs(cfg, key, default_key):
    dirs = cfg.get(key)
    if dirs is None:
        single = cfg.get(default_key)
        return [single] if single else []
    if isinstance(dirs, str):
        return [dirs]
    return list(dirs)


def _build_target_dataset(cls, split, base_cfg, root_dir):
    sub_cfg = copy.copy(base_cfg)
    sub_cfg['target_root_dir'] = root_dir
    sub_cfg['target_eval_root_dir'] = root_dir
    return cls(split=split, cfg=sub_cfg)


def _build_concat(cls, split, base_cfg, root_dirs):
    datasets = [_build_target_dataset(cls, split, base_cfg, d) for d in root_dirs]
    if len(datasets) == 1:
        return datasets[0]
    return CustomConcatDataset(datasets)


def build_dataloader(cfg, workers=4):
    if cfg['type'] != 'KITTI':
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    source_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
    test_source_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)

    train_root_dirs = _resolve_target_root_dirs(cfg, 'target_root_dirs', 'target_root_dir')
    eval_root_dirs = _resolve_target_root_dirs(cfg, 'target_eval_root_dirs', 'target_eval_root_dir')
    if not train_root_dirs:
        raise ValueError("No target training roots found: set 'target_root_dir' or 'target_root_dirs' in cfg.dataset")
    if not eval_root_dirs:
        eval_root_dirs = train_root_dirs

    target_set = _build_concat(TargetDataset, cfg['train_split'], cfg, train_root_dirs)
    test_target_set = _build_concat(TargetDataset, cfg['test_split'], cfg, eval_root_dirs)
    eval_target_set = _build_concat(TargetDatasetEval, cfg['test_split'], cfg, eval_root_dirs)

    test_source_loader = DataLoader(dataset=test_source_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    test_target_loader = DataLoader(dataset=test_target_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    eval_target_loader = DataLoader(dataset=eval_target_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    source_sampler = DistributedSampler(source_set) if dist.is_initialized() else None
    target_sampler = DistributedSampler(target_set) if dist.is_initialized() else None
    source_loader = DataLoader(dataset=source_set,
                               batch_size=cfg['batch_size'],
                               num_workers=workers,
                               worker_init_fn=my_worker_init_fn,
                               sampler=source_sampler,
                               shuffle=(source_sampler is None),
                               pin_memory=False,
                               drop_last=True,
                               persistent_workers=(workers > 0))
    target_loader = DataLoader(dataset=target_set,
                               batch_size=cfg['batch_size'],
                               num_workers=workers,
                               worker_init_fn=my_worker_init_fn,
                               sampler=target_sampler,
                               shuffle=(target_sampler is None),
                               pin_memory=False,
                               drop_last=True,
                               persistent_workers=(workers > 0))

    train_loader = [source_loader, target_loader]
    train_sampler = [source_sampler, target_sampler]
    test_loader = [test_source_loader, test_target_loader, eval_target_loader]

    return train_loader, test_loader, train_sampler
