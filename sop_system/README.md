# SOP 实时动作检测系统

基于 YOLO + MediaPipe + 时序模型 + FSM 的实时标准操作程序检测。

**动作序列**: 打开纸盒 → 放入耳机 → 放入插头 → 放入绿袋 → 关闭纸盒

## 两种运行方式

### 方式一：纯本地（推荐）

不需要服务器，直接在本机跑。有显卡用显卡，没显卡用 CPU。

```bash
cd sop_system
pip install -r requirements.txt

# USB 摄像头
python local_run.py --camera webcam

# 海康工业相机 (需装 MVS SDK)
python local_run.py --camera hik

# 视频文件
python local_run.py --video demo.mp4
```

操作：**空格** 开始，**R** 重置，**Q** 退出。

### 方式二：服务器推理

本地采集画面 → 发到 4090 服务器推理 → 返回结果。适合本地算力不够的场景。

**服务器端** (192.168.31.19):

```bash
ssh zhaowei@192.168.31.19
cd /home/zhaowei/shabi/sop_system
~/miniconda3/envs/three_env/bin/python server_inference/inference_server.py --stage 3
```

**本地客户端**:

```powershell
cd sop_system
set SOP_SERVER_HOST=192.168.31.19
python local_client/hik_stream_client.py
```

## 发给别人用

参见 [DEPLOY.md](DEPLOY.md)。三步搞定：

```bash
pip install -r requirements.txt
python tools/calibrate_rois.py   # 标定盒子位置
python local_run.py --camera webcam
```

## 项目结构

```
sop_system/
├── local_run.py                  # 本地运行入口
├── main.py                       # 旧版入口
├── run.py                        # 旧版海康相机入口
├── requirements.txt              # Python 依赖
├── DEPLOY.md                     # 部署说明
│
├── engine/                       # 核心检测引擎
│   ├── yolo_detector.py          # YOLO 目标检测 + ByteTrack
│   ├── hand_detector.py          # MediaPipe 手部 21 点关键点
│   ├── detection_stabilizer.py   # 检测结果防抖确认
│   ├── box_state_stabilizer.py   # 盒子开/关状态稳定
│   ├── event_detector.py         # 事件检测 (入盒/盒子开/关)
│   ├── action_segmenter.py       # 动作阶段分割
│   ├── object_state_tracker.py   # 物体状态追踪
│   ├── temporal_predictor_v2.py  # 时序动作识别 (BiGRU)
│   └── sop_fsm.py                # 有限状态机序列验证
│
├── models/                       # 模型文件 (已包含)
│   ├── yolo_final_v1.pt          # YOLO 目标检测
│   ├── hand_landmarker.task      # MediaPipe 手部关键点
│   └── temporal/                 # 时序模型
│
├── configs/
│   └── realcam_sop.yaml          # ROI 区域配置
│
├── server_inference/             # 服务器推理模式
│   └── inference_server.py       # WebSocket 推理服务器
│
├── local_client/                 # 服务器模式客户端
│   └── hik_stream_client.py     # 海康相机 → WebSocket 客户端
│
└── tools/                        # 工具
    └── calibrate_rois.py         # ROI 区域标定
```

## 检测原理

```
相机帧 → YOLO 目标检测 (5类)
       → MediaPipe 手部检测 (21点)
       → 物理状态引擎 (手-物-盒交互)
       → 时序模型 (动作识别)
       → 事件检测 (入盒/盒子开/关)
       → FSM 状态机 (序列验证 + 错误诊断)
       → 融合决策
       → 显示 + 报警
```

## 配置参数

主要参数在命令行指定，部分在代码中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | auto | cuda / cpu / auto |
| `--conf` | 0.12 | YOLO 置信度阈值 |
| `--imgsz` | 640 | YOLO 输入尺寸 |
| `--no-temporal` | false | 禁用时序模型 |
| `--no-display` | false | 无 GUI 模式 |

## 帧率参考

| 配置 | FPS |
|------|-----|
| RTX 4090 (服务器) | 25-40 |
| RTX 3060+ | 15-25 |
| RTX 3060+ 无时序 | 25-40 |
| CPU 无时序 | 3-8 |
