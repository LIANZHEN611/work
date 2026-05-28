import copy
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import argparse
import datetime
import yaml
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.tester_helper import Tester
from lib.helpers.trainer_helper import Trainer
from lib.helpers.utils_helper import create_logger, set_random_seed


def is_dist_launch():
    return 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ


def setup_dist():
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    distributed = is_dist_launch()
    if distributed:
        local_rank, rank = setup_dist()
    else:
        local_rank, rank = 0, 0
    try:
        _main(local_rank, rank, distributed)
    finally:
        cleanup_dist()


def _main(local_rank, rank, distributed):
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    camera_id = args.camera_id
    if args.threshold is not None:
        cfg['trainer']['threshold_increase_list'] = args.threshold
        if args.model_name is not None:
            cfg['model_name'] = args.model_name
            if rank == 0:
                print(cfg['model_name'])
        cfg['model']['model_name'] = cfg['model_name']
        if args.evaluate_only and camera_id is not None:
            cfg['dataset']['target_eval_root_dir'] = \
                "data/02_TUMTraffic-Synthetic/target_dataset/" + camera_id + "_dataset"

    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        timestamp = ''
    if distributed:
        obj = [timestamp]
        dist.broadcast_object_list(obj, src=0)
        timestamp = obj[0]
    run_name = f'{model_name}_{timestamp}'
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], run_name)
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
    if distributed:
        dist.barrier()

    log_file = None
    if rank == 0:
        log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file, rank=rank)

    train_loader, test_loader, train_sampler = build_dataloader(cfg['dataset'])
    if rank == 0:
        logger.info('Created train and test dataloader.')

    model, loss, matcher = build_model(cfg['model'])
    loss.dataloader = train_loader[1]
    if rank == 0:
        logger.info('Created prime model.')

    ema_model = copy.deepcopy(model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    if rank == 0:
        logger.info('Created ema model.')

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ema_model = ema_model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        gpu_ids = list(map(int, cfg['trainer'].get('gpu_ids', '0').split(',')))
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpu_ids)

    model_list = [model, ema_model]

    if args.evaluate_only:
        if rank == 0:
            logger.info('###################  Evaluation Only  ##################')
            tester = Tester(cfg=cfg['tester'],
                            model=model.module if hasattr(model, 'module') else model,
                            dataloader=test_loader,
                            logger=logger,
                            train_cfg=cfg['trainer'],
                            model_name=model_name)
            tester.set_output_folder(camera_id)
            tester.set_tester_mode('target')
            tester.test()
        return

    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    val_target_loader_for_trainer = test_loader[1]

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model_list,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      matcher_cfg=cfg['model'],
                      model_name=model_name,
                      val_target_loader=val_target_loader_for_trainer,
                      rank=rank,
                      train_sampler=train_sampler)
    trainer.output_dir = output_path

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.student_model.module if hasattr(trainer.student_model, 'module') else trainer.student_model,
                    dataloader=test_loader,
                    logger=logger,
                    loss=loss,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    tester.output_dir = output_path
    if cfg['dataset']['test_split'] != 'test' and rank == 0:
        trainer.tester = tester

    if rank == 0:
        logger.info('###################  Training  ##################')
        logger.info('Batch Size (per rank): %d' % (cfg['dataset']['batch_size']))
        logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if rank == 0 and cfg['dataset']['test_split'] != 'test':
        logger.info('###################  Testing  ##################')
        tester.set_tester_mode('target')
        tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', default="configs/monodetr.yaml",
                        help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('-threshold', type=float, help='Set the threshold for detection')
    parser.add_argument('-camera_id', type=str, help='Set the camera id for detection')
    parser.add_argument('-model_name', type=str, help='Set the model name for detection')
    args = parser.parse_args()
    main()
