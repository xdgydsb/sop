"""
SOP实时动作检测系统 — 配置文件
动作序列: S1开盒 → S2放耳机 → S3放插头 → S4放绿袋 → S5关盒
"""
import os
from pathlib import Path

# ===== 项目路径 =====
BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/home/zhaowei/shabi/data")
MODELS_DIR = Path("/home/zhaowei/shabi/sop_system/runs")

# ===== YOLO检测类别 (5类) =====
# box_open/box_closed分开: 因为开盒和关盒视觉差异大, 是两个不同状态
# 手用MediaPipe, 不用YOLO
CLASS_NAMES = ["box_closed", "box_open", "earphone", "charger", "green_bag"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_NAMES_CN = {
    "box_closed": "关闭的纸盒",
    "box_open":   "打开的纸盒",
    "earphone":   "黑色耳机盒",
    "charger":    "白色插头",
    "green_bag":  "绿色小袋子",
}

# ===== SOP步骤定义 =====
STEP_NAMES = {
    0: "等待开始",
    1: "1.打开纸盒",
    2: "2.放入耳机盒",
    3: "3.放入插头",
    4: "4.放入绿袋",
    5: "5.关闭纸盒",
    6: "完成",
}

# 步骤对应的目标物品(用于空间判断)
STEP_OBJECT = {2: "earphone", 3: "charger", 4: "green_bag"}
CORRECT_ITEM_ORDER = ["earphone", "charger", "green_bag"]

# ===== 合法状态转移 =====
STATE_TRANSITIONS = {
    0: [1],           # IDLE → S1
    1: [2],           # S1 → S2
    2: [3],           # S2 → S3
    3: [4],           # S3 → S4
    4: [5],           # S4 → S5
    5: [6],           # S5 → DONE
    6: [],            # DONE
    7: [0],           # ERROR → IDLE (可重置)
}

# ===== 物理状态检测参数 =====
BOX_OPEN_CONFIDENCE_MIN = 0.3      # 盒子被判定为open的最小置信度
OBJECT_IN_BOX_MARGIN = 15          # 物品在盒内判定的边距(像素)
HAND_BOX_PROXIMITY = 120           # 手与盒子接近的阈值(像素)
HAND_OBJECT_PROXIMITY = 80         # 手与物品交互的距离(像素)
TRACK_INIT_REGION_RADIUS = 60      # 物体初始区域判定半径(像素)
TRACK_MAX_LOST = 30                # 追踪最大丢失帧数
TRACK_TRAJECTORY_LEN = 30          # 轨迹历史长度(帧)

# ===== 防抖与确认参数 =====
CONFIRM_FRAMES = 8                 # 状态变化确认所需连续帧数
CONFIRM_RATIO = 0.7                # 确认缓冲区中目标占比阈值
MIN_STEP_DURATION = 0.4            # 最小步骤持续时间(秒), 防瞬间切换
STATE_TIMEOUT = 30.0               # 单步超时(秒)

# ===== EMA平滑参数 =====
EMA_ALPHA_BOX = 0.3                # 盒子状态 EMA 平滑系数
EMA_ALPHA_MODEL = 0.4              # 模型概率 EMA 平滑系数
EMA_ALPHA_KEYPOINTS = 0.5          # 手部关键点 EMA 平滑系数

# ===== 事件确认参数 (文档对齐) =====
BOX_OPEN_STABLE_FRAMES = 10        # 盒子打开确认所需稳定帧数
BOX_CLOSE_STABLE_FRAMES = 10       # 盒子关闭确认所需稳定帧数
OBJECT_IN_BOX_STABLE_FRAMES = 10   # 物体入盒稳定帧数
OBJECT_APPROACH_FRAMES = 3         # 手接近物体确认帧数
OBJECT_TOUCH_FRAMES = 2            # 手接触物体确认帧数
HAND_LEFT_AFTER_PLACE_FRAMES = 5   # 手离开后物体仍稳定帧数

# ===== LSTM时序模型参数 =====
LSTM_SEQ_LEN = 48                  # LSTM输入序列长度 (130维模型)
LSTM_SEQ_LEN_V2 = 32               # 时序模型输入序列长度 V2 (90维模型)
LSTM_INPUT_SIZE = 130              # 特征维度(完整版含双手关键点)
LSTM_INPUT_SIZE_V2 = 90            # 特征维度V2 (主动手+物体轨迹)
LSTM_INPUT_SIZE_REDUCED = 46       # 精简特征(去手部关键点)
LSTM_HIDDEN_SIZE = 128
LSTM_HIDDEN_SIZE_SMALL = 64        # 精简版LSTM隐藏层大小
LSTM_NUM_LAYERS = 2
LSTM_NUM_CLASSES = 7               # 0=背景, 1-5=步骤, 6=完成
LSTM_DROPOUT = 0.3
LSTM_DROPOUT_HIGH = 0.7            # 精简版高dropout防过拟合
MULTI_SCALE_WINDOWS = [32, 48, 64] # 多尺度窗口

# ===== 融合决策权重 =====
PHYSICAL_STATE_WEIGHT = 0.55       # 物理状态权重(硬约束)
LSTM_WEIGHT = 0.30                 # LSTM分类权重(软分类)
FSM_WEIGHT = 0.15                  # FSM约束权重

# ===== YOLO训练参数 =====
YOLO_MODEL_SIZE = "m"              # yolov8m
YOLO_EPOCHS = 100
YOLO_BATCH = 16
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.3
YOLO_IOU_THRESHOLD = 0.45

# ===== 视频参数 =====
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 25

# ===== 服务器配置 =====
# Sensitive values read from environment; fallbacks are empty/unset
SERVER_HOST = os.environ.get("SOP_SERVER_HOST", "")
SERVER_USER = os.environ.get("SOP_SERVER_USER", "")
SERVER_PASS = os.environ.get("SOP_SERVER_PASS", "")
SERVER_PORT = int(os.environ.get("SOP_SERVER_PORT", "22"))
SERVER_WORK_DIR = os.environ.get("SOP_SERVER_WORK_DIR", "/home/zhaowei/shabi/sop_system")
CONDA_ENV = os.environ.get("SOP_CONDA_ENV", "box_yolo")

# ===== GUI配置 =====
GUI_UPDATE_INTERVAL = 30           # GUI更新间隔(ms)
LOG_BUFFER_SIZE = 1000
ACTION_COLORS = {
    0: (128, 128, 128),    # IDLE - 灰色
    1: (0, 200, 255),      # S1 - 橙色
    2: (255, 100, 0),      # S2 - 蓝色
    3: (0, 255, 150),      # S3 - 绿色
    4: (200, 0, 200),      # S4 - 紫色
    5: (0, 150, 255),      # S5 - 深橙
}
ERROR_RED = (0, 0, 220)
SUCCESS_GREEN = (0, 200, 0)

# ===== 错误类型 =====
ERROR_TYPES = {
    "WRONG_ORDER":  "动作顺序错误",
    "MISSING_STEP": "遗漏步骤",
    "TIMEOUT":      "动作超时",
    "WRONG_OBJECT": "错误物品放入",
    "BOX_STATE_ERR":"纸盒状态异常",
    "NO_DETECTION": "未检测到物体",
}
