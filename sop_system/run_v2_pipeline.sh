#!/bin/bash
# SOP v2 全流程执行脚本 — 服务器端
# Usage: bash run_v2_pipeline.sh [step_number]
#   step_number: 1-7 (optional, runs all if omitted)

set -e
cd /home/zhaowei/shabi/520
export PYTHONPATH="/home/zhaowei/shabi/520:/home/zhaowei/shabi/sop_system:$PYTHONPATH"
PYTHON=python3

echo "=== SOP V2 Pipeline ==="
echo "Working dir: $(pwd)"
echo "Date: $(date)"
echo ""

run_step() {
    local step=$1
    case $step in
        1)
            echo "[Step 1] Convert manual npy to segments..."
            $PYTHON tools/convert_manual_npy_to_segments.py
            ;;
        2)
            echo "[Step 2] Extract 90-dim features (ALL videos)..."
            $PYTHON tools/extract_features_v2_90.py --video-type all --device cuda --imgsz 640
            ;;
        3)
            echo "[Step 3] Build frame-level labels..."
            $PYTHON tools/build_frame_labels_v2.py
            ;;
        4)
            echo "[Step 4] Build temporal windows..."
            $PYTHON tools/build_temporal_windows_v2.py
            ;;
        5)
            echo "[Step 5] Train TCN+BiGRU model..."
            $PYTHON train_temporal_v2.py --device cuda --epochs 80 --batch-size 64
            ;;
        6)
            echo "[Step 6] Evaluate model..."
            $PYTHON eval_temporal_v2.py --device cuda
            ;;
        7)
            echo "[Step 7] Pipeline-level evaluation..."
            $PYTHON eval_pipeline_offline_v2.py --device cuda
            ;;
        *)
            echo "Unknown step: $step"
            ;;
    esac
}

if [ $# -eq 0 ]; then
    for s in 1 2 3 4 5 6 7; do
        run_step $s
    done
else
    run_step $1
fi

echo ""
echo "=== Pipeline complete ==="
echo "Results:"
echo "  Segments: data/annotations/segment_annotations_v2.csv"
echo "  Features: data/features/v2_90/"
echo "  Labels:   data/labels/frame_labels_v2/"
echo "  Dataset:  data/datasets/temporal_v2_90_T48_S8/"
echo "  Model:    models/temporal/v2_90_tcn_bigru/"
echo "  Reports:  reports/temporal_v2_90/"
