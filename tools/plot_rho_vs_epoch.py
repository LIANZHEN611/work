import argparse
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_XTICKS = 40


def extract_rho_series(run_md_path: str):
    with open(run_md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    section = re.search(r"###\s*3\.3.*?```(.*?)```", text, flags=re.S)
    if not section:
        raise ValueError("未在 run.md 中找到 '3.3 实际训练观察到的 ρ 轨迹' 代码块。")

    block = section.group(1)
    values = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", block)]
    if not values:
        raise ValueError("未在 ρ 轨迹代码块中解析到数值。")
    return values


def plot_rho(values, out_path: str, title: str, max_xticks: int):
    epochs = list(range(1, len(values) + 1))
    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, values, marker='o', linewidth=1.8, markersize=4, color='#1f77b4')
    plt.xlabel('Epoch')
    plt.ylabel(r'$\rho$ (keep_rate)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    step = 1 if len(epochs) <= max_xticks else max(1, len(epochs) // max_xticks)
    plt.xticks(epochs[::step])
    plt.ylim(min(values) - 0.005, max(values) + 0.005)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='根据 run.md 中的日志数据绘制 ρ 随 epoch 变化折线图')
    parser.add_argument('--run-md', default='run.md', help='run.md 文件路径')
    parser.add_argument('--out', default='vis/rho_vs_epoch.png', help='输出图片路径')
    parser.add_argument('--title', default='Adap-EMA: ρ vs Epoch', help='图标题')
    parser.add_argument('--max-xticks', type=int, default=MAX_XTICKS, help='x 轴最多显示多少个刻度')
    args = parser.parse_args()

    values = extract_rho_series(args.run_md)
    plot_rho(values, args.out, args.title, args.max_xticks)
    print(f'parsed {len(values)} rho values from {args.run_md}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
