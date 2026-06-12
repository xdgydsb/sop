# v2_90_tcn_bigru_center_T48 — 版本说明

## 版本标识

| 属性 | 值 |
|------|-----|
| **版本名** | v2_90_tcn_bigru_center_T48 |
| **冻结日期** | 2025-05-20 |
| **状态** | FROZEN — 离线基线，不再覆盖 |

## 模型

| 属性 | 值 |
|------|-----|
| 架构 | TCN (3 blocks, dilations=1,2,4, k=3) + BiGRU (2 layers, hidden=128, bidirectional) |
| 参数量 | 904,079 |
| 输入维度 | 90 |
| 输出 | step(6类), verb(4类), target(5类) |
| 窗口长度 T | 48 帧 (~3.2s @ 15fps) |
| 标签帧位置 | center-frame (frame 24) |
| 模型路径 | `models/temporal/v2_90_tcn_bigru/checkpoints/best.pt` |
| 训练配置 | `models/temporal/v2_90_tcn_bigru/train_config.yaml` |

## 数据

| 属性 | 值 |
|------|------|
| 特征 fps | 15.0 (np.linspace 均匀采样) |
| 总视频数 | 244 (181 OK + 63 WR, wr_60 缺标注) |
| 总窗口数 | 5,942 |
| 划分方式 | 按 video_name, 70/15/15 (train/val/test) |
| 特征路径 | `data/features/v2_90/` |
| 标签路径 | `data/labels/frame_labels_v2/` |
| 数据集路径 | `data/datasets/temporal_v2_90_T48_S8/` |

## 离线评估指标

### 模型级 (Test集 947窗口, 38视频)

| 指标 | 值 |
|------|-----|
| Step Accuracy | **97.15%** |
| Verb Accuracy | 98.10% |
| Target Accuracy | 97.04% |
| Best Epoch | 20 (早停于 35) |

### 各类别 F1

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| idle | 0.792 | 0.905 | 0.844 |
| S1 开盒 | 0.989 | 0.978 | 0.983 |
| S2 放黑(耳机) | 0.977 | 0.969 | 0.973 |
| S3 放白(插头) | 1.000 | 0.927 | 0.962 |
| S4 放绿(袋子) | 0.919 | 0.988 | 0.952 |
| S5 关盒 | 0.991 | 0.982 | 0.986 |

### S3/S4 混淆

- S3→S4: 8/123 (6.5%)
- S4→S3: 0/161 (0%)
- 总混淆率: 2.8%

### 管线级 (全量 244 视频, FSM)

| 指标 | 值 |
|------|-----|
| OK 通过率 | **179/181 (98.9%)** |
| WR 检出率 | **63/63 (100.0%)** |
| 平均帧准确率 | 87.6% |

### 误报 OK

- ok_24: S3/S4 边界混淆, FSM 纠错
- ok_65: S3/S4 边界混淆, FSM 纠错

## 已知局限

1. S3(白色插头) vs S4(绿色袋子) 边界混淆 2.8%，手部动作相似，颜色特征区分度不足
2. 事件检测 precision 偏低 (23-44%)，EMA 平滑导致边界模糊 — 时序分类器固有特性
3. wr_60 缺人工标注，无法参与训练
4. 完全依赖手部+物体特征，极端遮挡下性能下降
5. 模型为 center-frame 输出，实时推理约 1.6s 延迟

## 文件清单

- 模型: `models/temporal/v2_90_tcn_bigru/checkpoints/best.pt` (11MB)
- 报告: `reports/temporal_v2_90/`
  - `test_metrics.json`, `fsm_eval.json`, `event_sequence_eval.json`
  - `confusion_step.png`, `confusion_target.png`, `confusion_s3_s4.json`
  - `bad_cases.json`
- 特征: `data/features/v2_90/{ok,wr}/*.npz` (245 files)
- 标签: `data/labels/frame_labels_v2/{ok,wr}/*.npz` (244 files)
- 数据集: `data/datasets/temporal_v2_90_T48_S8/`

## 后续版本规划

| 版本 | 内容 | 触发条件 |
|------|------|----------|
| v2.1_event_guard | 增强 PhysicalStateEngine/EventDetector 物理判定 | 回放暴露事件触发不稳 |
| v3 | 特征增强(颜色直方图/attention)、重新训练 | v2.1 仍无法解决 S3/S4 混淆 |
