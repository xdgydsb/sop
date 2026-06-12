# SOP 实时动作检测系统 — 部署说明

## 环境要求

- Python 3.9+
- 推荐: NVIDIA GPU + CUDA 11.8+ (CPU 也能跑，帧率较低)
- 海康 MVS SDK (仅连接海康工业相机时需要)

## 快速开始

### 1. 一键安装（推荐）

双击 `setup.bat`，自动检测显卡并安装对应版本的 PyTorch + 所有依赖。

### 2. 或手动安装

```bash
cd sop_system
pip install -r requirements.txt
```

### 3. 标定 ROI (换机位必须做)

盒子在画面中的区域因摄像头位置而异。首次运行前标定一次：

```bash
python tools/calibrate_rois.py
```

按提示框出盒子区域即可，标定结果保存到 `configs/realcam_sop.yaml`。
不标定也能跑，但盒子检测可能不准。

### 3. 运行

**USB 摄像头 / 笔记本自带摄像头:**
```bash
python local_run.py --camera webcam
```

**海康工业相机 (需要 MVS SDK):**
```bash
# 如果 MVS 装在非默认路径，设置环境变量:
set MVS_ROOT=D:\你的路径\MVS

python local_run.py --camera hik
```

MVS SDK 默认会按以下顺序自动查找:
1. `MVS_ROOT` 环境变量
2. `D:\develop\MVS`
3. `C:\Program Files (x86)\MVS`
4. `C:\Program Files\MVS`

**视频文件:**
```bash
python local_run.py --video 视频路径.mp4
```

**纯 CPU 模式 (无 GPU):**
```bash
python local_run.py --camera webcam --device cpu --no-temporal
```

### 3. 操作说明

| 按键 | 功能 |
|------|------|
| **空格** | 开始 SOP 检测 (PREVIEW → ARMED → RUNNING) |
| **R** | 重置状态 (回到 PREVIEW) |
| **Q** / **ESC** | 退出 |

### 4. 检测流程

```
PREVIEW → 按空格 → ARMED → 打开纸盒 → RUNNING → 依次放物品 → COMPLETE
```

进度条颜色:
- 灰色 = 未完成
- 绿色 = 已完成
- 青色闪烁 = 正在进行
- 红色 = 错误

## 目录结构

```
sop_system/
├── local_run.py               # 本地运行入口 ← 主要用这个
├── main.py                    # 旧版入口 (训练/检测)
├── run.py                     # 旧版海康相机入口
│
├── engine/                    # 核心引擎
│   ├── yolo_detector.py       # YOLO 目标检测
│   ├── hand_detector.py       # MediaPipe 手部检测
│   ├── detection_stabilizer.py # 检测稳定器
│   ├── box_state_stabilizer.py # 盒子状态稳定器
│   ├── event_detector.py      # 事件检测
│   ├── sop_fsm.py             # 状态机
│   └── ...
│
├── models/                    # 模型文件 (已包含，无需额外下载)
│   ├── yolo_final_v1.pt       # YOLO 目标检测模型
│   ├── hand_landmarker.task   # MediaPipe 手部关键点模型
│   └── temporal/              # 时序动作识别模型
│
├── configs/
│   └── realcam_sop.yaml       # ROI 区域配置 (不同摄像头需要重新标定)
│
├── server_inference/          # 服务器推理模式 (4090 远程推理)
├── tools/                     # 标定/调试工具
│   └── calibrate_rois.py      # ROI 区域标定工具
└── requirements.txt           # Python 依赖
```

## 帧率说明

| 配置 | 预期帧率 |
|------|---------|
| RTX 3060+ + 时序模型 | 15-25 FPS |
| RTX 3060+ 无时序 (`--no-temporal`) | 25-40 FPS |
| CPU 无时序 | 3-8 FPS |
| CPU + 时序模型 | 1-3 FPS |

## 常见问题

**Q: 摄像头没画面?**
检查摄像头是否被其他程序占用，或尝试 `--webcam-id 1`

**Q: box 检测框位置不对?**
运行 ROI 标定工具重新设定盒子区域:
```bash
python tools/calibrate_rois.py
```

**Q: 海康相机报 "MVS SDK 未安装"?**
1. 确认 MVS 已安装
2. 设置 `MVS_ROOT` 环境变量指向安装目录
3. 或者用 `--camera webcam` 切换到 USB 摄像头

**Q: `ModuleNotFoundError: No module named 'torch'`?**
```bash
pip install -r requirements.txt
```

**Q: MediaPipe 报错?**
项目自动适配新旧 MediaPipe API，确保版本 ≥ 0.10.30:
```bash
pip install "mediapipe>=0.10.30"
```
