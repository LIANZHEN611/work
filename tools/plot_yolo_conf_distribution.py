import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts')
for _fn in sorted(os.listdir(_FONTS_DIR)) if os.path.isdir(_FONTS_DIR) else []:
    if _fn.lower().endswith(('.ttf', '.otf', '.ttc')):
        try:
            font_manager.fontManager.addfont(os.path.join(_FONTS_DIR, _fn))
        except Exception:
            pass
matplotlib.rcParams['axes.unicode_minus'] = False

def _fp(name):
    p = os.path.join(_FONTS_DIR, name)
    return FontProperties(fname=p) if os.path.isfile(p) else FontProperties()

FONT_HEAVY = _fp('PingFang Heavy.ttf')
FONT_BOLD = _fp('PingFang Bold.ttf')
FONT_MEDIUM = _fp('PingFang Medium.ttf')
FONT_REGULAR = _fp('PingFang Regular.ttf')
FONT_LIGHT = _fp('PingFang Light.ttf')
matplotlib.rcParams['font.family'] = FONT_REGULAR.get_name()
import matplotlib.pyplot as plt


DATA = 'data/02_TUMTraffic-Synthetic/target_dataset'
CAMERAS = ['s110_n', 's110_o', 's110_s', 's110_w']
CAM_NAME = {'s110_n': '北侧', 's110_o': '东侧', 's110_s': '南侧', 's110_w': '西侧'}
COCO_TO_PROJECT = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 3: 'Cyclist', 5: 'Bus', 7: 'BigCar'}
CAM_COLOR = {'s110_n': '#1f77b4', 's110_o': '#ff7f0e', 's110_s': '#2ca02c', 's110_w': '#d62728'}
CLASS_COLOR = {'Car': '#00bfff', 'BigCar': '#d62728', 'Bus': '#2ca02c',
               'Pedestrian': '#ff7f0e', 'Cyclist': '#ffd700'}


def collect_confs(data_root, cameras, yolo_subdirs):
    per_cam = {}
    per_cls = {}
    for cam in cameras:
        confs = []
        for sub in yolo_subdirs:
            ydir = os.path.join(data_root, f'{cam}_dataset', 'training', sub)
            if not os.path.isdir(ydir):
                continue
            for fn in os.listdir(ydir):
                if not fn.endswith('.txt'):
                    continue
                with open(os.path.join(ydir, fn)) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) < 6:
                            continue
                        try:
                            cid = int(p[0])
                            conf = float(p[5])
                        except ValueError:
                            continue
                        if cid not in COCO_TO_PROJECT:
                            continue
                        cls = COCO_TO_PROJECT[cid]
                        confs.append(conf)
                        per_cls.setdefault(cls, []).append(conf)
            break
        per_cam[cam] = np.array(confs)
    return per_cam, per_cls


def print_summary(per_cam, per_cls):
    print('=== YOLO 2D detection confidence distribution ===')
    head = f'{"camera":>10} | {"count":>9} | {"mean":>6} | {"median":>6} | {"<0.25":>6} | {"<0.40":>6} | {"<0.50":>6} | {"<0.70":>6}'
    print(head); print('-' * len(head))
    all_confs = []
    for cam in CAMERAS:
        c = per_cam.get(cam, np.array([]))
        if len(c) == 0:
            continue
        all_confs.append(c)
        print(f'{cam:>10} | {len(c):>9} | {c.mean():>6.3f} | {np.median(c):>6.3f} | '
              f'{(c < 0.25).mean()*100:>5.1f}% | {(c < 0.40).mean()*100:>5.1f}% | '
              f'{(c < 0.50).mean()*100:>5.1f}% | {(c < 0.70).mean()*100:>5.1f}%')
    if all_confs:
        overall = np.concatenate(all_confs)
        print(f'{"overall":>10} | {len(overall):>9} | {overall.mean():>6.3f} | {np.median(overall):>6.3f} | '
              f'{(overall < 0.25).mean()*100:>5.1f}% | {(overall < 0.40).mean()*100:>5.1f}% | '
              f'{(overall < 0.50).mean()*100:>5.1f}% | {(overall < 0.70).mean()*100:>5.1f}%')
    print()
    print('=== Per-class (across cameras) ===')
    head = f'{"class":>11} | {"count":>9} | {"mean":>6} | {"median":>6} | {"<0.25":>6} | {"<0.40":>6}'
    print(head); print('-' * len(head))
    for cls in ['Car', 'BigCar', 'Bus', 'Pedestrian', 'Cyclist']:
        c = np.array(per_cls.get(cls, []))
        if len(c) == 0:
            print(f'{cls:>11} | {0:>9} |   --   |   --   |   --   |   --  ')
            continue
        print(f'{cls:>11} | {len(c):>9} | {c.mean():>6.3f} | {np.median(c):>6.3f} | '
              f'{(c < 0.25).mean()*100:>5.1f}% | {(c < 0.40).mean()*100:>5.1f}%')


def render(per_cam, per_cls, out_path, conf_thr_start, conf_thr_end, xlo, xhi, bins):
    bin_edges = np.linspace(xlo, xhi, bins + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_cam_grid = axes[:, :2].ravel()
    ax_cdf_cam = axes[0, 2]
    ax_cdf_cls = axes[1, 2]

    def _style_ticks(ax):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(FONT_LIGHT)
            lbl.set_fontsize(9)

    for ax, cam in zip(ax_cam_grid, CAMERAS):
        c = per_cam.get(cam, np.array([]))
        if len(c) == 0:
            ax.set_visible(False)
            continue
        ax.hist(c, bins=bin_edges, color=CAM_COLOR[cam], edgecolor='white', linewidth=0.5)
        ax.axvline(conf_thr_end, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.axvline(c.mean(), color='black', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.set_title(f'S110 {CAM_NAME[cam]}  '
                     f'(样本数={len(c)}, 均值={c.mean():.3f}, 中位数={np.median(c):.3f})',
                     fontproperties=FONT_BOLD, fontsize=13)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel('YOLO 置信度', fontproperties=FONT_REGULAR, fontsize=11)
        ax.set_ylabel('样本数', fontproperties=FONT_REGULAR, fontsize=11)
        ax.grid(alpha=0.3)
        ax.text(conf_thr_end + 0.01, ax.get_ylim()[1] * 0.92,
                f'终止阈值={conf_thr_end:.2f}\n(低于占 {(c < conf_thr_end).mean()*100:.1f}%)',
                fontproperties=FONT_LIGHT, fontsize=9, color='red', va='top')
        _style_ticks(ax)

    for cam in CAMERAS:
        c = np.sort(per_cam.get(cam, np.array([])))
        if len(c) == 0:
            continue
        cdf = np.arange(1, len(c) + 1) / len(c)
        ax_cdf_cam.plot(c, cdf, label=CAM_NAME[cam], color=CAM_COLOR[cam], linewidth=2)
    ax_cdf_cam.axvline(conf_thr_end, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                       label=f'终止阈值={conf_thr_end:.2f}')
    ax_cdf_cam.set_xlim(xlo, xhi); ax_cdf_cam.set_ylim(0, 1.0)
    ax_cdf_cam.set_xlabel('YOLO 置信度', fontproperties=FONT_REGULAR, fontsize=11)
    ax_cdf_cam.set_ylabel('置信度 $\\leq$ x 的累积比例', fontproperties=FONT_REGULAR, fontsize=11)
    ax_cdf_cam.set_title('按相机的累积分布', fontproperties=FONT_BOLD, fontsize=13)
    leg = ax_cdf_cam.legend(loc='lower right', prop=FONT_LIGHT)
    for txt in leg.get_texts(): txt.set_fontsize(10)
    ax_cdf_cam.grid(alpha=0.3); _style_ticks(ax_cdf_cam)

    CLASS_CN = {'Car': '小车', 'BigCar': '大车', 'Bus': '公交车',
                'Pedestrian': '行人', 'Cyclist': '骑车人'}
    for cls in ['Car', 'BigCar', 'Bus', 'Pedestrian', 'Cyclist']:
        c = np.sort(np.array(per_cls.get(cls, [])))
        if len(c) == 0:
            continue
        cdf = np.arange(1, len(c) + 1) / len(c)
        ax_cdf_cls.plot(c, cdf, label=f'{CLASS_CN[cls]} (n={len(c)})',
                        color=CLASS_COLOR[cls], linewidth=2)
    ax_cdf_cls.axvline(conf_thr_end, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_cdf_cls.set_xlim(xlo, xhi); ax_cdf_cls.set_ylim(0, 1.0)
    ax_cdf_cls.set_xlabel('YOLO 置信度', fontproperties=FONT_REGULAR, fontsize=11)
    ax_cdf_cls.set_ylabel('置信度 $\\leq$ x 的累积比例', fontproperties=FONT_REGULAR, fontsize=11)
    ax_cdf_cls.set_title('按类别的累积分布（全相机汇总）', fontproperties=FONT_BOLD, fontsize=13)
    leg = ax_cdf_cls.legend(loc='lower right', prop=FONT_LIGHT)
    for txt in leg.get_texts(): txt.set_fontsize(10)
    ax_cdf_cls.grid(alpha=0.3); _style_ticks(ax_cdf_cls)

    plt.suptitle('目标域验证集 YOLO 2D 检测置信度', fontproperties=FONT_HEAVY, fontsize=18, y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=DATA)
    p.add_argument('--source', default='image_2_yolo,image_2_yolo12',
                   help='comma-separated YOLO label subdir candidates; first match per cam wins')
    p.add_argument('--out', default='vis/yolo_conf_distribution.png')
    p.add_argument('--conf-thr-start', type=float, default=0.05)
    p.add_argument('--conf-thr-end', type=float, default=0.40)
    p.add_argument('--xlo', type=float, default=0.2)
    p.add_argument('--xhi', type=float, default=1.0)
    p.add_argument('--bins', type=int, default=32)
    args = p.parse_args()

    subdirs = [f'{s.strip()}/labels' if not s.endswith('/labels') else s.strip()
               for s in args.source.split(',')]
    per_cam, per_cls = collect_confs(args.data_root, CAMERAS, subdirs)
    if not any(len(v) for v in per_cam.values()):
        print(f'no YOLO labels found under {args.data_root}/<cam>_dataset/training/{subdirs}',
              file=sys.stderr)
        return
    print_summary(per_cam, per_cls)
    render(per_cam, per_cls, args.out, args.conf_thr_start, args.conf_thr_end,
           args.xlo, args.xhi, args.bins)


if __name__ == '__main__':
    main()

