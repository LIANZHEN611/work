# 项目中两项机制的代码实现梳理与伪代码

## 1) 自适应 EMA 动量更新（Adaptive EMA Keep Rate）

### 代码位置
- `lib/helpers/trainer_helper.py`
  - 初始化参数：`__init__` 中 `adap_ema / keep_rate_min / keep_rate_max / adap_ema_alpha / _current_keep_rate / _last_depth_loss`
  - EMA 参数融合：`ema_update(self, keep_rate)`
  - 训练中使用与自适应更新：`train_one_epoch`（`batch_idx % ema_step == 0` 触发 EMA；epoch 末根据深度损失调整 keep_rate）
  - 验证信号：`compute_val_depth_loss`（可选）

### 实现方法（按代码逻辑）
1. 初始化当前 EMA 动量系数 `current_keep_rate`（来自配置 `keep_rate`）。
2. 每 `ema_step` 个 batch 执行一次 teacher 参数更新：
   - 若开启 `adap_ema`，使用 `current_keep_rate`
   - 否则使用固定 `keep_rate`
3. epoch 结束后（仅 `adap_ema=true`）：
   - 计算当前深度损失信号 `cur`（优先 val，若无则用 train 平均）
   - 与上一次损失 `last_depth_loss` 比较，算相对改善率  
     `rel = (last - cur) / (last + 1e-6)`
   - 若 `rel>0`（损失下降），减小 keep_rate（更快跟随 student）
   - 若 `rel<=0`（损失变差或不变），增大 keep_rate（更平滑更稳）
   - keep_rate 始终裁剪在 `[keep_rate_min, keep_rate_max]`

### 伪代码
```text
初始化:
  current_keep_rate = cfg.keep_rate
  last_depth_loss = None

函数 EMA_UPDATE(student, teacher, keep_rate):
  对 teacher 每个浮点参数 key:
    teacher[key] = (1 - keep_rate) * student[key] + keep_rate * teacher[key]

每个 epoch:
  depth_loss_sum = 0, depth_loss_n = 0
  对每个 target batch:
    ...训练 student...
    if batch_idx % ema_step == 0:
      kr = current_keep_rate if adap_ema else cfg.keep_rate
      EMA_UPDATE(student, teacher, kr)
    记录 loss_depth 到 depth_loss_sum/depth_loss_n

  if adap_ema:
    cur = compute_val_depth_loss()，若无则 cur = depth_loss_sum / depth_loss_n
    if cur 有效:
      if last_depth_loss 有效且 > 0:
        rel = (last_depth_loss - cur) / (last_depth_loss + 1e-6)
        if rel > 0:
          current_keep_rate = max(keep_rate_min, current_keep_rate - alpha * rel)
        else:
          current_keep_rate = min(keep_rate_max, current_keep_rate + alpha * (-rel))
      last_depth_loss = cur
```

---

## 2) 置信度重加权与动态阈值

### 代码位置
- `lib/helpers/trainer_helper.py`
  - 配置读取：`_conf_reweight / _conf_thr_start / _conf_thr_end / _current_conf_threshold`
  - 伪标签置信度重加权：`extract_dets_from_outputs`（`if self._conf_reweight:`）
  - 动态阈值更新：`train_one_epoch`  
    `frac = epoch / (max_epoch-1)` 线性插值到 `_current_conf_threshold`
  - 按阈值过滤伪标签：`train_on_target_with_pseudo_labels` 与 `compute_val_depth_loss`
- `lib/models/monodetr/matcher.py`
  - `filter_high_cost_pairs_and_sort(..., cost_threshold)`：按匹配代价阈值保留匹配对（由 `threshold_increase_list` 提供）

### 实现方法（按代码逻辑）
1. **置信度来源重加权**
   - 教师模型输出分类 logits，取 `sigmoid` 后每 query 的最大类别概率作为 `teacher_conf`。
   - 若样本存在 YOLO 2D 置信度（`has_yolo_conf`），则优先使用 YOLO 置信度；否则回退到 `teacher_conf`。
   - 得到统一的伪标签置信度 `targets_cpu['conf']`。
2. **动态阈值**
   - 每个 epoch 开始按线性调度更新阈值  
     `current_conf_threshold = start + (end-start) * frac`，`frac∈[0,1]`
   - 在目标域训练和验证损失计算前，过滤伪标签：仅保留 `conf >= current_conf_threshold` 且几何有效（深度/3D框尺寸有效）的目标。
3. **匹配代价阈值过滤**
   - 在 Hungarian 匹配后，`filter_high_cost_pairs_and_sort` 用 `threshold_increase_list` 过滤代价过高的匹配，仅保留可靠对。

### 伪代码
```text
函数 BUILD_PSEUDO_CONF(outputs, targets_cpu):
  teacher_conf = max(sigmoid(outputs.pred_logits), dim=class)
  teacher_conf = 按匹配索引重排

  if 有 yolo_conf 且有 has_yolo_conf:
    has_yolo_sample = (sum(has_yolo_conf, dim=1) > 0)
    conf = has_yolo_sample * yolo_conf + (1 - has_yolo_sample) * teacher_conf
  else:
    conf = teacher_conf
  return conf

每个 epoch 开始:
  frac = min(1, epoch / (max_epoch - 1))
  current_conf_threshold = conf_thr_start + (conf_thr_end - conf_thr_start) * frac

每个 target batch:
  outputs_teacher = teacher(weak_aug_input)
  indices, cost_map = hungarian_match(outputs_teacher, targets)
  sorted_indices, filtered_indices, mask = filter_high_cost_pairs_and_sort(indices, cost_map, threshold_increase_list)
  pseudo_targets = 从 outputs_teacher 提取并按 sorted_indices 对齐
  pseudo_targets.conf = BUILD_PSEUDO_CONF(outputs_teacher, targets_cpu)

  对每张图的伪标签 t:
    valid = depth_valid AND 3d_box_size_valid
    if conf_reweight:
      valid = valid AND (t.conf >= current_conf_threshold)
    t = t[valid]
```

---

## 3) 最终整体算法（项目中这两项机制与训练流程的集成）

```text
输入:
  source_loader, target_loader, (可选) val_target_loader
  student_model, teacher_model
  参数: keep_rate, adap_ema*, conf_reweight*, conf_threshold_start/end, threshold_increase_list

阶段A: Burn-in
  仅用 source 真值监督训练 student 若干 epoch
  burn-in 后执行一次 EMA_UPDATE(keep_rate=0)，将 teacher 同步到 student

阶段B: 主训练 (epoch=0..max_epoch-1)
  若启用 conf_reweight:
    动态更新 current_conf_threshold (线性从 start 到 end)

  对每个 target batch:
    1) teacher 在弱增强图像上推理
    2) Hungarian 匹配 + 代价阈值过滤(threshold_increase_list)
    3) 生成伪标签(3D框/深度/朝向等)
    4) 若 conf_reweight:
         生成 conf (YOLO conf 优先, 否则 teacher conf)
         按 current_conf_threshold 过滤低置信伪标签
    5) 用过滤后的伪标签监督 student 反向传播更新
    6) 每 ema_step 执行一次 EMA_UPDATE:
         keep_rate = current_keep_rate (adap_ema) 或固定 keep_rate

  epoch 结束:
    若 adap_ema:
      计算当前深度损失信号(优先 val, 否则 train 平均)
      与 last_depth_loss 比较，更新 current_keep_rate 到 [min,max]

输出:
  student 持续学习目标域
  teacher 通过 EMA 平滑跟踪 student
  伪标签质量由“代价阈值 + 置信度重加权 + 动态置信阈值”共同控制
```

