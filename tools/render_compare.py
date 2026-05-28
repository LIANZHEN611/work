import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


DATA = 'data/02_TUMTraffic-Synthetic/target_dataset'
PRED_BASE = 'outputs/eval_aligned'
BASELINE_TAG = 'baseline_ep10'
OPTIM_TAG = 'optim_ep39'

ALL_DATASETS = ['s110_n', 's110_o', 's110_s', 's110_w']
DATASET_NAME = {'s110_n': 'North', 's110_o': 'East', 's110_s': 'South', 's110_w': 'West'}

COCO_TO_PROJECT = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 3: 'Cyclist', 5: 'Bus', 7: 'BigCar'}
KITTI_CLASS_NORMALIZE = {
    'Truck': 'BigCar', 'Trailer': 'BigCar', 'Motorcycle': 'Cyclist',
    'TRUCK': 'BigCar', 'TRAILER': 'BigCar', 'MOTORCYCLE': 'Cyclist',
    'BIGCAR': 'BigCar', 'PEDESTRIAN': 'Pedestrian', 'CYCLIST': 'Cyclist',
    'CAR': 'Car', 'VAN': 'Van', 'BUS': 'Bus',
}
CLASS_PRIORITY = {'BigCar': 0.99, 'Bus': 0.96, 'Van': 0.93, 'Car': 0.90, 'Cyclist': 0.87, 'Pedestrian': 0.84}
CLASS_COLOR = {
    'Car': '#00bfff', 'Van': '#1f77b4', 'Bus': '#2ca02c', 'BigCar': '#d62728',
    'Pedestrian': '#ff7f0e', 'Cyclist': '#ffd700',
}

NMS_IOU_BY_CLASS = {'Car': 0.5, 'Van': 0.5, 'Bus': 0.5, 'BigCar': 0.4, 'Pedestrian': 0.3, 'Cyclist': 0.4}
NMS_IOU_DEFAULT = 0.5
CROSS_CLASS_VEHICLE_IOU = 0.65
CROSS_CLASS_PERSON_IOU = 0.25
ANY_CLASS_IOU = 0.55
VEHICLE_CLASSES = {'Car', 'Van', 'Bus', 'BigCar'}
PERSON_CLASSES = {'Pedestrian', 'Cyclist'}
SCORE_THR = 0.25

CELL_FS = 18
ROW_FS = 22
FIG_FS = 28
LABEL_FS = 7
COL_WSPACE = 0.05
ROW_HSPACE = 0.30
TITLE_OFFSET = 0.006

IMG_W_PX, IMG_H_PX = 1920, 1200
ASPECT = IMG_W_PX / IMG_H_PX


def figsize_for(n_rows, n_cols=3, fig_w=24.0, wspace=COL_WSPACE):
    per_col_w = fig_w / (n_cols + (n_cols - 1) * wspace)
    per_col_h = per_col_w / ASPECT
    fig_h = n_rows * per_col_h + 1.1 * n_rows + 0.6
    return (fig_w, fig_h)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 1e-6 else 0.0


def nms(dets):
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[0], []).append(d)
    kept = []
    for cls, items in by_cls.items():
        items.sort(key=lambda d: -(d[2] or 0))
        thr = NMS_IOU_BY_CLASS.get(cls, NMS_IOU_DEFAULT)
        survivors = []
        for cand in items:
            if all(iou_xyxy(cand[1], s[1]) <= thr for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    kept.sort(key=lambda d: -(d[2] or 0))
    final = []
    for cand in kept:
        c = cand[0]
        suppress = False
        for s in final:
            sc = s[0]
            iou = iou_xyxy(cand[1], s[1])
            if sc != c:
                if c in VEHICLE_CLASSES and sc in VEHICLE_CLASSES and iou > CROSS_CLASS_VEHICLE_IOU:
                    suppress = True
                    break
                if c in PERSON_CLASSES and sc in PERSON_CLASSES and iou > CROSS_CLASS_PERSON_IOU:
                    suppress = True
                    break
            if iou > ANY_CLASS_IOU:
                suppress = True
                break
        if not suppress:
            final.append(cand)
    return final


def parse_kitti(path, is_pred):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8:
                continue
            try:
                x1, y1, x2, y2 = map(float, p[4:8])
            except ValueError:
                continue
            score = None
            if is_pred and len(p) >= 16:
                try:
                    score = float(p[-1])
                except ValueError:
                    score = None
            cls = KITTI_CLASS_NORMALIZE.get(p[0], p[0])
            if not is_pred and score is None:
                score = CLASS_PRIORITY.get(cls, 0.5)
            out.append((cls, (x1, y1, x2, y2), score))
    return out


def parse_yolo(path, W, H):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                cid = int(p[0])
                cx = float(p[1]) * W
                cy = float(p[2]) * H
                w = float(p[3]) * W
                h = float(p[4]) * H
                conf = float(p[5]) if len(p) >= 6 else None
            except ValueError:
                continue
            proj = COCO_TO_PROJECT.get(cid)
            if proj is None:
                continue
            out.append((proj, (cx - w/2, cy - h/2, cx + w/2, cy + h/2), conf))
    return out


def pick_yolo_files(cam, skip, k):
    ydir = f'{DATA}/{cam}_dataset/training/image_2_yolo/labels'
    cands = []
    if os.path.isdir(ydir):
        for fn in os.listdir(ydir):
            if not fn.endswith('.txt'):
                continue
            with open(os.path.join(ydir, fn)) as f:
                n = sum(1 for line in f if line.strip())
            if n >= 3:
                cands.append((n, fn))
    cands.sort(reverse=True)
    return [fn for _, fn in cands[skip:skip + k]]


def pick_kitti_files(cam, skip, k):
    ldir = f'{DATA}/{cam}_dataset/training/label_2'
    cands = []
    for fn in os.listdir(ldir):
        if not fn.endswith('.txt'):
            continue
        sz = os.path.getsize(os.path.join(ldir, fn))
        if sz > 100:
            cands.append((sz, fn))
    cands.sort(reverse=True)
    return [fn for _, fn in cands[skip:skip + k]]


def add_cell_label(ax, label):
    ax.text(0.5, -0.03, label, transform=ax.transAxes,
            ha='center', va='top', fontsize=CELL_FS)
    ax.axis('off')


def draw_boxes(ax, img, dets, cell_title, score_thr=0.0, hide_score=False):
    ax.imshow(img)
    for cls, (x1, y1, x2, y2), score in dets:
        if score is not None and score < score_thr:
            continue
        c = CLASS_COLOR.get(cls, '#ffffff')
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor=c, facecolor='none'))
        if hide_score or score is None:
            label = cls
        else:
            label = f'{cls} {score:.2f}'
        ax.text(x1, max(y1 - 4, 4), label, color=c, fontsize=LABEL_FS,
                bbox=dict(facecolor='black', alpha=0.45, pad=0.6, edgecolor='none'))
    add_cell_label(ax, cell_title)


def render_grid(cam, files, gt_source, ydir, name, out_path):
    n_rows = len(files)
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize_for(n_rows),
                             gridspec_kw={'hspace': ROW_HSPACE, 'wspace': COL_WSPACE})
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for i, fn in enumerate(files):
        stem = fn[:-4]
        img = Image.open(f'{DATA}/{cam}_dataset/training/image_2/{stem}.jpg')
        W, H = img.size

        if gt_source == 'yolo':
            gt = parse_yolo(f'{ydir}/{fn}', W, H)
            gt = [d for d in gt if d[2] is None or d[2] >= SCORE_THR]
            gt = nms(gt)
            gt_hide_score = False
        else:
            gt = parse_kitti(f'{DATA}/{cam}_dataset/training/label_2/{fn}', is_pred=False)
            gt = nms(gt)
            gt_hide_score = True

        base = parse_kitti(f'{PRED_BASE}/{BASELINE_TAG}_{cam}/preds/{fn}', is_pred=True)
        opt = parse_kitti(f'{PRED_BASE}/{OPTIM_TAG}_{cam}/preds/{fn}', is_pred=True)
        base_nms = nms([d for d in base if d[2] is None or d[2] >= SCORE_THR])
        opt_nms = nms([d for d in opt if d[2] is None or d[2] >= SCORE_THR])

        draw_boxes(axes[i, 0], img, gt, 'GT', hide_score=gt_hide_score)
        draw_boxes(axes[i, 1], img, base_nms, 'Baseline')
        draw_boxes(axes[i, 2], img, opt_nms, 'Optimized')

    fig.canvas.draw()
    top_left_pos = axes[0, 0].get_position()
    fig.text(top_left_pos.x0, top_left_pos.y1 + 0.025, f'S110 {name}',
             ha='left', va='bottom', fontsize=FIG_FS, fontweight='bold')
    for i, fn in enumerate(files):
        stem = fn[:-4]
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 2].get_position()
        center_x = (pos_left.x0 + pos_right.x1) / 2
        top_y = pos_left.y1
        fig.text(center_x, top_y + TITLE_OFFSET, f'Frame #{stem}',
                 ha='center', va='bottom', fontsize=ROW_FS)

    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}  (frames={[fn[:-4] for fn in files]})')


def render_dataset(cam, num_images, frames_per_image, skip_top_k, out_dir):
    name = DATASET_NAME[cam]
    total_frames = num_images * frames_per_image
    ydir = f'{DATA}/{cam}_dataset/training/image_2_yolo/labels'
    if os.path.isdir(ydir):
        files = pick_yolo_files(cam, skip=skip_top_k, k=total_frames)
        gt_source = 'yolo'
    else:
        files = pick_kitti_files(cam, skip=skip_top_k, k=total_frames)
        gt_source = 'label_2'
    if not files:
        print(f'no GT files found for {cam}', file=sys.stderr)
        return

    for g in range(num_images):
        group = files[g * frames_per_image:(g + 1) * frames_per_image]
        if not group:
            break
        out_path = os.path.join(out_dir, f'compare_{cam}_{g+1:02d}.png')
        render_grid(cam, group, gt_source, ydir, name, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all',
                   choices=ALL_DATASETS + ['all'],
                   help='which s110 camera to render (default: all 4)')
    p.add_argument('--num-images', type=int, default=8,
                   help='number of output PNGs per dataset')
    p.add_argument('--frames-per-image', type=int, default=3,
                   help='rows per output PNG (frames stacked vertically per file)')
    p.add_argument('--skip-top-k', type=int, default=8,
                   help='skip the top-K most-annotated frames; frames are picked from rank skip:skip+num_frames')
    p.add_argument('--out-dir', default='vis')
    p.add_argument('--no-clean', action='store_true',
                   help='do not delete existing compare_*.png in out-dir before rendering')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_clean:
        for f in glob.glob(os.path.join(args.out_dir, 'compare_*.png')):
            os.remove(f)
            print(f'removed {f}')

    cameras = ALL_DATASETS if args.dataset == 'all' else [args.dataset]
    for cam in cameras:
        render_dataset(cam, args.num_images, args.frames_per_image, args.skip_top_k, args.out_dir)


if __name__ == '__main__':
    main()
