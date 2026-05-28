# Visualization & Analysis Tools

## 1. `tools/render_compare.py` — 三栏对比图（GT / Baseline / Optimized）

为每个 s110 相机生成 N 张对比图，每张图包含 K 帧（每帧 3 个面板：GT、Baseline、Optimized）。

### 用法

```bash
python tools/render_compare.py [--dataset {s110_n,s110_o,s110_s,s110_w,all}] \
                               [--num-images N] [--frames-per-image K] \
                               [--skip-top-k S] [--out-dir vis] [--no-clean]
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset` | `all` | 单相机或 `all`（4 个相机一起）|
| `--num-images` | `8` | 每个 dataset 输出 PNG 数量 |
| `--frames-per-image` | `3` | 每张 PNG 包含的帧数（行数）|
| `--skip-top-k` | `8` | 跳过最密集标注的前 K 帧，从第 K+1 开始抽样 |
| `--out-dir` | `vis` | 输出目录 |
| `--no-clean` | flag | 不删除已有 `compare_*.png`（默认会清空）|

### 默认产出

```
4 datasets × 8 images × 3 frames = 32 PNGs / 96 frames total
vis/compare_s110_n_01.png ... vis/compare_s110_n_08.png
vis/compare_s110_o_01.png ... vis/compare_s110_o_08.png
vis/compare_s110_s_01.png ... vis/compare_s110_s_08.png
vis/compare_s110_w_01.png ... vis/compare_s110_w_08.png
```

每张 PNG：N rows × 3 cols，每行一帧。每帧三列依次是 YOLO GT、Baseline 预测、Optimized 预测。所有面板都跑了 per-class + cross-class (vehicle/person) + class-agnostic 大重叠抑制。

### 常用示例

```bash
# 默认全 4 相机, 每相机 8 张 PNG × 3 帧
python tools/render_compare.py

# 只渲染南相机, 5 张图, 每张 4 帧
python tools/render_compare.py --dataset s110_s --num-images 5 --frames-per-image 4

# 跳过更多前排帧, 拿到下一批
python tools/render_compare.py --skip-top-k 32

# 保留已有 PNG, 增量生成
python tools/render_compare.py --no-clean --dataset s110_w
```

### 模型 checkpoint 来源

脚本顶部硬编码了：
- `BASELINE_TAG = 'baseline_ep10'` → 读 `outputs/eval_aligned/baseline_ep10_<cam>/preds/`
- `OPTIM_TAG = 'optim_ep39'` → 读 `outputs/eval_aligned/optim_ep39_<cam>/preds/`

要换 checkpoint 时直接改这两行。预测目录由 `tools/eval_aligned.py` 在 eval 时产出。

---

## 2. `tools/plot_yolo_conf_distribution.py` — YOLO 置信度分布

统计目标域 YOLO 伪标签的 2D 检测置信度分布，输出 2×3 子图（4 个相机的直方图 + 2 张 CDF）。

### 用法

```bash
python tools/plot_yolo_conf_distribution.py [--source <subdirs>] [--out PATH] \
                                            [--conf-thr-start 0.05] [--conf-thr-end 0.40] \
                                            [--xlo 0.2] [--xhi 1.0] [--bins 32]
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--source` | `image_2_yolo,image_2_yolo12` | 标签子目录候选, 按顺序匹配第一个存在的 |
| `--out` | `vis/yolo_conf_distribution.png` | 输出 PNG 路径 |
| `--conf-thr-start` | `0.05` | 配置里的起始过滤阈值（仅用于参考线/注记）|
| `--conf-thr-end` | `0.40` | 配置里的最终过滤阈值（图上红线）|
| `--xlo` / `--xhi` | `0.2 / 1.0` | x 轴范围 |
| `--bins` | `32` | 直方图箱数 |

```bash
# 默认（用原始 YOLO 标签）
python tools/plot_yolo_conf_distribution.py

# 模拟不同 conf 阈值
python tools/plot_yolo_conf_distribution.py --conf-thr-end 0.25

# 用 YOLO12 重标的伪标签
python tools/plot_yolo_conf_distribution.py --source image_2_yolo12
```

---

## 3. EMA 动量系数（Adap-EMA 配置）

### 3.1 配置项（`configs/monodetr_optim.yaml`）

| 参数 | 值 | 含义 |
|---|---|---|
| `ema_update_frequency` | 1 | EMA 每 epoch 更新（外层频率）|
| `ema_step` | 50 | 每 50 个 batch 内层 ratchet 一次 |
| `keep_rate` | 0.95 | EMA 保留率 ρ 的**初始值** |
| `keep_rate_min` | 0.90 | ρ **下限**（loss 快速下降时教师可加速）|
| `keep_rate_max` | 0.99 | ρ **上限**（loss 上升时教师减速保稳）|
| `adap_ema_alpha` | 0.05 | 每个 epoch 对 ρ 的调整步长 |
| `adap_ema_val_batches` | 50 | val pass 采样 batch 数（提供 ρ 调整信号）|
| `adap_ema` | true | 自适应 EMA 开关 |

**baseline (`monodetr.yaml`)** 没有 `adap_ema`，ρ 固定 0.95 全程不变。

### 3.2 调整规则

每 epoch 结束跑 val pass，得到当前 epoch 的目标域 `loss_depth`。然后：

```
rel = (last_depth_loss - cur_depth_loss) / (last_depth_loss + 1e-6)
if rel > 0:   # loss 下降 → 教师可加速 → 减小 ρ
    ρ = max(0.90, ρ - 0.05 * rel)
else:         # loss 上升 → 教师减速保稳 → 增大 ρ
    ρ = min(0.99, ρ + 0.05 * (-rel))
```

### 3.3 实际训练观察到的 ρ 轨迹（`outputs/train_optim_studentonly_*.log`）

```
early epoch: 0.9900 → 0.9789 → 0.9869 → 0.9900 → 0.9848 → 0.9869 → 0.9900
mid   epoch: 0.9844 → 0.9772 → 0.9765 → 0.9855 → 0.9800 → 0.9764 → 0.9867 → 0.9776
late  epoch: 0.9877 → 0.9835 → 0.9894 → 0.9857 → 0.9871 → 0.9860 → 0.9850 → 0.9896 → 0.9900 → 0.9882
```

- ρ 从 0.95 起步，在 early epoch val loss 大幅波动时迅速被推到 0.99 上限
- 之后稳定在 **0.97–0.99** 区间震荡，**很少触及下限 0.90**
- 后期更靠近 0.99，说明教师被有意减慢以稳定后期训练
- 一阶 EMA 时间常数 `≈ 1/(1-ρ)`：0.95 ≈ 20 步, 0.98 ≈ 50 步 — 教师平均比 baseline 慢约 2-4 倍

---

## 4. 目标数据集验证集 2D 检测置信度分布

s110 各相机的 val 集 == train 集（同一份图像），YOLO 伪标签的置信度分布即是模型实际接触到的训练信号。

### 4.1 按相机汇总

```
camera   | count   | mean  | median | <0.25 | <0.40 | <0.50 | <0.70
s110_n   |  8115   | 0.762 | 0.840  | 0.0%  |  6.3% | 12.7% | 28.1%
s110_o   | 12139   | 0.733 | 0.786  | 0.0%  |  8.8% | 15.4% | 34.6%
s110_s   | 28292   | 0.631 | 0.641  | 0.0%  | 13.9% | 28.7% | 59.6%
s110_w   | 11448   | 0.633 | 0.648  | 0.0%  | 14.1% | 27.2% | 59.2%
overall  | 59994   | 0.670 | 0.698  | 0.0%  | 11.9% | 23.6% | 50.2%
```

### 4.2 按类别汇总（全 4 相机）

```
class      | count   | mean  | median | <0.25 | <0.40
Car        | 40973   | 0.667 | 0.698  | 0.0%  | 11.6%
BigCar     | 10985   | 0.697 | 0.737  | 0.0%  | 10.0%
Bus        |  2775   | 0.751 | 0.863  | 0.0%  |  9.7%
Pedestrian |  4834   | 0.591 | 0.613  | 0.0%  | 18.6%
Cyclist    |   427   | 0.552 | 0.540  | 0.0%  | 21.3%
```

### 4.3 可视化（`vis/yolo_conf_distribution.png`）

- **左上 4 格**：每个相机一张独立直方图（已不再重叠）
- **右上**：按相机的 CDF — 一眼看出"阈值 X 会砍掉多少 label"
- **右下**：按类别的 CDF — 直接对比哪类受阈值影响最大

### 4.4 关键观察

1. **置信度下限就是 0.25**：YOLO 上游已按 0.25 过滤，所以配置里的 `conf_threshold_start=0.05` 实际上**全程无效**（没有 conf<0.25 的样本可过滤）。
2. **`conf_threshold_end=0.40` 整体砍掉约 12% 的伪标签**，但**类别失衡严重**：
   - 车辆类（Car / BigCar / Bus）：mean ≈ 0.67–0.75，过滤损失 10–12%（温和）
   - **Pedestrian**：mean 0.59，被过滤 **18.6%**
   - **Cyclist**：mean 0.55，被过滤 **21.3%**，且原本样本量就只有 427（5 类中最稀有）
3. **相机间差异显著**：
   - North / East 是右偏分布（high-conf 主导），mean 0.73–0.76
   - South / West 接近均匀分布，mean 仅 0.63，约 60% 的 label conf < 0.7
   - South/West 受阈值过滤影响最严重 — 与 eval 中 optim 在这两个相机的 mAP regression 最明显完全吻合
4. **改进方向**：把 `conf_threshold_end` 从 0.40 降到 0.25–0.30，或者采用 **per-class 阈值**（车辆类 0.4，行人/骑车人 0.15–0.20）可以同时保留 Conf-Reweight 对噪声的抑制效果，并避免对小/稀有类的过度过滤。
