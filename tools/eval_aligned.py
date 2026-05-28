import argparse
import copy
import datetime
import os
import sys
import time

import numpy as np
import torch
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from torch.utils.data import DataLoader

from lib.datasets.kitti.target_dataset_eval import TargetDatasetEval
from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import create_logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--aligned-root',
                   default='data/02_TUMTraffic-Synthetic/target_dataset/tum_traffic_intersection_kitti_aligned')
    p.add_argument('--split', default='val')
    p.add_argument('--out', default='outputs/eval_aligned')
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--tag', default=None)
    args = p.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    ds_cfg = copy.deepcopy(cfg['dataset'])
    ds_cfg['target_eval_root_dir'] = args.aligned_root
    ds_cfg['test_split'] = args.split

    tag = args.tag or os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'eval.%s.log' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    logger.info(f'Config: {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')
    logger.info(f'Aligned root: {args.aligned_root}')
    logger.info(f'Split: {args.split}')

    dataset = TargetDatasetEval(split=args.split, cfg=ds_cfg)
    logger.info(f'Dataset size: {len(dataset)}')
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    model, _, _ = build_model(cfg['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    load_checkpoint(model=model, optimizer=None, filename=args.checkpoint,
                    map_location=device, logger=logger)
    model.eval()

    results = {}
    infer_t = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, calibs, targets, info = batch
            inputs = inputs.to(device)
            calibs = calibs.to(device)
            img_sizes = info['img_size'].to(device)
            t0 = time.time()
            outputs = model(inputs, calibs, targets, img_sizes)
            infer_t += time.time() - t0
            dets = extract_dets_from_outputs(outputs=outputs, K=dataset.max_objs, topk=args.topk)
            dets = dets.detach().cpu().numpy()
            calibs_list = [dataset.get_calib(int(idx)) for idx in info['img_id']]
            info_np = {k: v.detach().cpu().numpy() for k, v in info.items()}
            dets = decode_detections(dets=dets, info=info_np, calibs=calibs_list,
                                     cls_mean_size=dataset.cls_mean_size,
                                     threshold=args.threshold)
            results.update(dets)
            if batch_idx % 10 == 0:
                logger.info(f'batch {batch_idx}/{len(loader)} done')
    logger.info(f'Inference time per image: {infer_t / max(len(dataset), 1):.4f}s')

    results_dir = os.path.join(out_dir, 'preds')
    os.makedirs(results_dir, exist_ok=True)
    skipped = 0
    for img_id, dets_list in results.items():
        fpath = os.path.join(results_dir, f'{int(img_id):06d}.txt')
        with open(fpath, 'w') as f:
            for row in dets_list:
                if len(row) < 16:
                    skipped += 1
                    continue
                arr = np.asarray(row, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    skipped += 1
                    continue
                cls_idx = int(row[0])
                if cls_idx < 0 or cls_idx >= len(dataset.class_name):
                    skipped += 1
                    continue
                cn = dataset.class_name[cls_idx]
                f.write(f'{cn} 0.0 0')
                for j in range(1, len(row)):
                    f.write(f' {row[j]:.2f}')
                f.write('\n')
    logger.info(f'Wrote predictions for {len(results)} images (skipped {skipped} bad rows) to {results_dir}')

    gt_dir = os.path.join(args.aligned_root,
                          'testing' if args.split == 'test' else 'training',
                          'label_2')
    car_moderate = dataset.eval(results_dir=results_dir, gt_dir=gt_dir, logger=logger)
    logger.info(f'>>> CAR moderate mAP3D@R40 = {car_moderate}')
    print(f'\nFinal car_moderate mAP3D@R40 (aligned val): {car_moderate}\n')


if __name__ == '__main__':
    main()
