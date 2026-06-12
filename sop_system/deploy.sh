#!/bin/bash
# SOP系统一键部署脚本
# 部署到4090服务器: zhaowei@192.168.31.19

set -e

SERVER="zhaowei@192.168.31.19"
REMOTE_DIR="/home/zhaowei/shabi/sop_system"
DATA_DIR="/home/zhaowei/shabi/data"
PASS="666888"

echo "========================================="
echo "SOP实时检测系统 - 一键部署"
echo "========================================="

# Step 1: 安装服务端依赖
echo ""
echo "[1/6] 安装服务器依赖..."
ssh "$SERVER" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate box_yolo && pip install ultralytics opencv-python mediapipe numpy pandas scikit-learn tqdm matplotlib Pillow pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -3"

# Step 2: 同步代码
echo ""
echo "[2/6] 同步代码到服务器..."
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'runs/' \
    /d/shabi/sop_system/ "$SERVER:$REMOTE_DIR/"

# Step 3: 检查GPU
echo ""
echo "[3/6] 检测GPU..."
ssh "$SERVER" "nvidia-smi | head -15"

# Step 4: 准备YOLO数据(抽帧+Grounding DINO标注)
echo ""
echo "[4/6] 准备YOLO训练数据..."
ssh "$SERVER" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate box_yolo && \
    cd $REMOTE_DIR && \
    python training/prepare_yolo_data.py \
        --data-dir $DATA_DIR \
        --output-dir $DATA_DIR/yolo_dataset_v3 \
        --ok-videos 80 --wr-videos 30 \
        --yuanzi-frames 60 --ok-frames 25 \
        --auto-annotate"

# Step 5: 训练YOLO
echo ""
echo "[5/6] 训练YOLOv8m..."
ssh "$SERVER" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate box_yolo && \
    cd $REMOTE_DIR && \
    python training/train_yolo.py \
        --data $DATA_DIR/yolo_dataset_v3/dataset.yaml \
        --model m --epochs 100 --batch 16 --device 0 \
        --name sop_yolo_v1"

# Step 6: 验证YOLO
echo ""
echo "[6/6] 验证YOLO检测效果..."
# 下载测试图片到本地
scp "$SERVER:$REMOTE_DIR/runs/train/sop_yolo_v1/results.png" /d/shabi/sop_system/runs/yolo_results.png 2>/dev/null || true

echo ""
echo "========================================="
echo "部署完成!"
echo "YOLO模型: $SERVER:$REMOTE_DIR/runs/train/sop_yolo_v1/weights/best.pt"
echo ""
echo "下一步:"
echo "1. 验证YOLO检测质量: 查看 runs/yolo_results.png"
echo "2. 提取特征: python training/extract_features.py --yolo-model runs/train/sop_yolo_v1/weights/best.pt"
echo "3. 训练LSTM: python training/train_lstm.py"
echo "4. 实时检测: python main.py --mode camera"
echo "========================================="
