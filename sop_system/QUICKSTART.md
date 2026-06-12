# SOP实时检测系统 - 快速启动指南

## 项目概述

实时检测5个动作序列:
1. **S1**: 打开纸盒
2. **S2**: 放入耳机盒
3. **S3**: 放入插头
4. **S4**: 放入绿袋
5. **S5**: 关闭纸盒

## 目录结构

```
sop_system/
├── config.py              # 配置
├── main.py                # 主程序
├── sop_detector.py        # 检测器核心
├── requirements.txt       # 依赖
├── test_system.py         # 系统测试
├── train_on_server.py     # 服务器训练脚本
│
├── models/                # 检测模型
├── utils/                 # 工具
├── training/              # 训练脚本
├── gui/                   # GUI界面
└── data/                  # 数据目录 (需要创建)
    ├── yuanzi/            # 5个独立动作视频
    ├── ok/                # 正确流程视频
    └── wr/                # 错误流程视频
```

## 快速开始

### 第1步: 安装依赖

```bash
cd sop_system
pip install -r requirements.txt
```

### 第2步: 测试系统

```bash
python test_system.py
```

### 第3步: 使用服务器训练模型 (推荐)

#### 3.1 部署到服务器

```bash
# 运行部署脚本 (在本地Windows)
# 或使用Git Bash/WSL
bash deploy_server.sh
```

#### 3.2 上传数据到服务器

```bash
# 上传数据
scp -r data/yuanzi/* zhaowei@192.168.31.19:/home/zhaowei/shabi/sop_system/data/yuanzi/
scp -r data/ok/* zhaowei@192.168.31.19:/home/zhaowei/shabi/sop_system/data/ok/
```

#### 3.3 SSH登录并训练

```bash
# SSH登录
ssh zhaowei@192.168.31.19

cd /home/zhaowei/shabi/sop_system

# 执行完整训练流程
python train_on_server.py --all --yolo-epochs 100 --temporal-epochs 50
```

#### 3.4 下载训练好的模型

```bash
# 在本地执行
scp zhaowei@192.168.31.19:/home/zhaowei/shabi/sop_system/models/trained/*.pt ./models/
```

### 第4步: 运行检测

#### 检测视频文件

```bash
python main.py detect \
    --source data/ok/test.avi \
    --yolo-model models/yolo_best.pt \
    --temporal-model models/temporal_best.pt \
    --output result.mp4
```

#### 海康相机实时检测

```bash
# 确保MVS SDK已安装
python main.py detect \
    --source camera \
    --device 0 \
    --yolo-model models/yolo_best.pt \
    --temporal-model models/temporal_best.pt
```

#### 启动GUI

```bash
python main.py gui
```

## 在4090服务器上的完整流程

```bash
# 1. 登录服务器
ssh zhaowei@192.168.31.19

# 2. 进入项目目录
mkdir -p ~/shabi/sop_system
cd ~/shabi/sop_system

# 3. 上传代码后，安装依赖
pip install -r requirements.txt

# 4. 准备数据 (如果数据已在服务器)
python main.py prepare-data --yuanzi data/yuanzi --ok data/ok

# 5. 训练YOLO模型 (约30分钟-1小时)
python main.py train-yolo \
    --data data/processed/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --model m \
    --device 0

# 6. 训练时序模型 (约10-20分钟)
python main.py train-temporal \
    --data data \
    --epochs 50 \
    --batch-size 8 \
    --device cuda

# 7. 复制模型
mkdir -p models/trained
cp runs/train/*/weights/best.pt models/trained/yolo_best.pt
cp runs/temporal/best_temporal_model.pt models/trained/temporal_best.pt

# 8. 测试检测
python main.py detect --source data/ok/test.avi
```

## 本地训练 (无服务器)

如果没有服务器，可以使用较小的模型:

```bash
# YOLO (使用nano模型更快)
python main.py train-yolo --data data/processed/dataset.yaml --epochs 50 --model n --device cpu

# 时序模型
python main.py train-temporal --data data --epochs 30 --device cpu
```

## 常见问题

### Q: 海康SDK找不到?
**A:** 安装MVS SDK并设置PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/opt/MVS/Samples/Python  # Linux
set PYTHONPATH=%PYTHONPATH%;C:\Program Files (x86)\MVS\Development\Samples\Python  # Windows
```

### Q: CUDA内存不足?
**A:** 减小batch_size:
```bash
python main.py train-yolo --batch 4  # 默认是8
```

### Q: YOLO检测框漂移?
**A:** 调整跟踪参数:
```python
# 在yolo_detector.py中
ObjectTracker(max_age=20)  # 增加跟踪寿命
```

### Q: 时序识别不准确?
**A:** 增加训练数据量，或调整时序窗口:
```python
# 在config.py中
TEMPORAL_WINDOWS = [48, 96, 128]  # 增加窗口大小
```

## 系统要求

- **最低配置**: CPU i5, 8GB RAM, 无GPU (仅测试)
- **推荐配置**: CPU i7, 16GB RAM, RTX 3060+
- **最佳配置**: CPU i9, 32GB RAM, RTX 4090 (服务器配置)

## 下一步

1. 运行 `python test_system.py` 验证环境
2. 准备数据并使用服务器训练模型
3. 测试视频检测效果
4. 连接海康相机进行实时检测

有问题查看 README.md 获取详细信息。
