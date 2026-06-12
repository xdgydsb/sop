"""
v2 管线路径配置 — server/local 自动切换
"""
import os
from pathlib import Path

# Detect environment
ON_SERVER = os.path.exists("/home/zhaowei")

if ON_SERVER:
    ROOT = Path("/home/zhaowei/shabi/520")
    SOP_SYSTEM = Path("/home/zhaowei/shabi/sop_system")
    RAW_VIDEOS_DIR = Path("/home/zhaowei/shabi/data")
    MANUAL_NPY_SRC = Path("/home/zhaowei/shabi/data/features_v4")
    YOLO_MODEL = SOP_SYSTEM / "models" / "yolo_final_v1.pt"
    DEVICE = "cuda"
    IMGSZ = 640
else:
    ROOT = Path(__file__).parent.parent  # sop_system/
    SOP_SYSTEM = ROOT
    RAW_VIDEOS_DIR = ROOT.parent / "data"
    MANUAL_NPY_SRC = RAW_VIDEOS_DIR / "features_v4"
    YOLO_MODEL = ROOT / "models" / "yolo_final_v1.pt"
    DEVICE = "cpu"
    IMGSZ = 480

# Output directories
ANNOTATIONS_DIR = ROOT / "data" / "annotations"
SEGMENT_CSV = ANNOTATIONS_DIR / "segment_annotations_v2.csv"
SEGMENT_JSON = ANNOTATIONS_DIR / "segment_annotations_v2.json"
MANUAL_NPY_DIR = ANNOTATIONS_DIR / "manual_npy"
CHECK_DIR = ANNOTATIONS_DIR / "label_check_reports"

FEAT_OUT_DIR = ROOT / "data" / "features" / "v2_90"
LABEL_OUT_DIR = ROOT / "data" / "labels" / "frame_labels_v2"
DATASET_DIR = ROOT / "data" / "datasets" / "temporal_v2_90_T48_S8"

MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"
LOGS_DIR = ROOT / "logs" / "temporal_v2_90"

# Feature config
FEATURE_DIM = 90
FEATURE_FPS = 15.0
EMA_ALPHA = 0.5
HOLD_FRAMES = 5

# Window config
T = 48
STRIDE = 8

# Ensure dirs exist
for d in [ANNOTATIONS_DIR, CHECK_DIR, FEAT_OUT_DIR, LABEL_OUT_DIR,
          DATASET_DIR, MODEL_DIR, CHECKPOINT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Config: {'SERVER' if ON_SERVER else 'LOCAL'}, device={DEVICE}")
print(f"Root: {ROOT}")
