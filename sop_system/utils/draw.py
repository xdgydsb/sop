"""可视化绘制 — YOLO检测框 + 步骤信息 + 错误提示"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 颜色定义
BOX_COLORS = {
    "box_closed": (128, 128, 128),   # 灰色
    "box_open":   (0, 220, 0),       # 绿色
    "earphone":   (255, 80, 80),     # 红色(黑耳机)
    "charger":    (255, 200, 80),    # 橙色(白插头)
    "green_bag":  (80, 255, 80),     # 亮绿
}
STEP_COLORS = {
    0: (128, 128, 128),
    1: (0, 200, 255),
    2: (255, 120, 0),
    3: (0, 255, 150),
    4: (180, 0, 220),
    5: (0, 150, 255),
}
STEP_NAMES = {
    0: "等待", 1: "S1 开盒", 2: "S2 放耳机",
    3: "S3 放插头", 4: "S4 放绿袋", 5: "S5 关盒",
}


def _load_font(size: int = 24):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """在帧上绘制YOLO检测框"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = BOX_COLORS.get(det.cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name}:{det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1-5, 15)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def draw_hand(frame: np.ndarray, hands: list) -> np.ndarray:
    """绘制手部检测"""
    for hand in hands:
        hx1, hy1, hx2, hy2 = hand.bbox
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 100), 2)
        for lm in hand.landmarks:
            px = int(lm[0] * frame.shape[1])
            py = int(lm[1] * frame.shape[0])
            cv2.circle(frame, (px, py), 2, (0, 220, 100), -1)
    return frame


def draw_step_bar(frame: np.ndarray, current_step: int, step_name: str,
                  progress: float, is_error: bool = False):
    """绘制底部步骤进度条和信息"""
    h, w = frame.shape[:2]

    # 顶部状态栏
    bar_color = (0, 0, 200) if is_error else (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 60), bar_color, -1)
    cv2.rectangle(frame, (0, 60), (w, 62), (80, 80, 80), -1)

    if is_error:
        cv2.putText(frame, "错误!", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, step_name[:60], (120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    else:
        if current_step == 0:
            status_text = "等待开始操作"
        elif current_step == 6:
            status_text = "完成!"
        else:
            status_text = f"步骤 {current_step}/5: {step_name}"
        cv2.putText(frame, status_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 底部进度条
    bar_y = h - 40
    bar_w = 300
    bar_x = (w - bar_w) // 2

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (50, 50, 50), -1)
    filled = int(bar_w * progress)
    p_color = (0, 220, 0) if progress >= 1.0 else (0, 200, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 18), p_color, -1)
    cv2.putText(frame, f"{current_step}/5",
                (bar_x + bar_w//2 - 20, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 步骤指示器 (右侧)
    for i in range(1, 6):
        sx = w - 250 + (i-1) * 50
        sy = bar_y + 4
        c = (0, 220, 0) if i <= current_step else (80, 80, 80)
        cv2.circle(frame, (sx + 8, sy + 8), 6, c, -1)
        cv2.putText(frame, f"S{i}", (sx - 3, sy - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    return frame


def draw_info_panel(frame: np.ndarray, result: dict):
    """绘制右侧信息面板"""
    h, w = frame.shape[:2]
    panel_x = w - 240
    panel_w = 230

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 65), (panel_x + panel_w, h - 55),
                  (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 100
    dy = 28
    font_scale = 0.5

    # YOLO检测结果
    cv2.putText(frame, "--- YOLO 检测 ---", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += dy
    detections = result.get("detections", {})
    for name, det in detections.items():
        if det:
            text = f"{name}: {det['conf']:.2f}"
            cv2.putText(frame, text, (panel_x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 220, 180), 1)
            y += dy - 4

    # 物理状态
    y += 10
    cv2.putText(frame, "--- 物理状态 ---", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += dy
    cv2.putText(frame, f"盒子: {result.get('box_state', '?')}", (panel_x + 15, y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 220, 180), 1)
    y += dy - 4
    items = result.get("objects_in_box", [])
    cv2.putText(frame, f"盒内物品: {items}", (panel_x + 15, y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 220, 180), 1)

    # LSTM结果
    y += 16
    cv2.putText(frame, "--- LSTM 识别 ---", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += dy
    lstm_step = result.get("lstm_step", 0)
    lstm_conf = result.get("lstm_conf", 0)
    cv2.putText(frame, f"动作: S{lstm_step} ({lstm_conf:.0%})",
                (panel_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 220, 180), 1)

    # Top3
    y += dy - 4
    top3 = result.get("lstm_top3", [])
    for rank, (step, conf) in enumerate(top3[:3]):
        sname = STEP_NAMES.get(step, f"S{step}")
        cv2.putText(frame, f"  {rank+1}. {sname}: {conf:.0%}",
                    (panel_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 180, 150), 1)
        y += 18

    return frame
