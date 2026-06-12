"""
SOP实时检测 — 相机画面优先，模型后台加载
"""
import sys, os, ctypes, time, cv2, numpy as np, threading, traceback, torch
from pathlib import Path
from collections import Counter

# SDK
_MVS = r"D:\develop\MVS\Development\Samples\Python"
sys.path.insert(0, _MVS)
sys.path.insert(0, os.path.join(_MVS, "MvImport"))
from MvCameraControl_class import *
from CameraParams_header import *
from MvErrorDefine_const import *

script_dir = Path(__file__).parent
CRASH_LOG = script_dir / "crash_log.txt"

# ═══════════════════════════════════
# 相机
# ═══════════════════════════════════

def init_camera():
    MvCamera.MV_CC_Initialize()
    dl = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dl)
    if dl.nDeviceNum == 0: raise RuntimeError("No camera")

    st = cast(dl.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    cam.MV_CC_CreateHandle(st)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetEnumValue("ExposureAuto", 2)
    cam.MV_CC_SetEnumValue("GainAuto", 2)

    stp = MVCC_INTVALUE()
    memset(byref(stp), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stp); ps = stp.nCurValue
    cam.MV_CC_GetIntValue("Width", stp); w = stp.nCurValue
    cam.MV_CC_GetIntValue("Height", stp); h = stp.nCurValue

    cam.MV_CC_StartGrabbing()
    buf = (c_ubyte * ps)()
    print(f"Camera: {w}x{h} payload={ps}")
    return cam, buf, ps, w, h


def read_frame(cam, buf, ps):
    info = MV_FRAME_OUT_INFO_EX()
    memset(byref(info), 0, sizeof(info))
    ret = cam.MV_CC_GetOneFrameTimeout(buf, ps, info, 1000)
    if ret != 0: return None

    w, h, fmt = info.nWidth, info.nHeight, info.enPixelType
    if fmt == 0x02180014:  # RGB8
        img = np.asarray(buf, dtype=np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == 0x02180015:  # BGR8
        return np.asarray(buf, dtype=np.uint8).reshape((h, w, 3))
    elif fmt == 0x01080001:  # Mono
        img = np.asarray(buf, dtype=np.uint8).reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif fmt in (0x01080009, 0x01080008, 0x0108000A, 0x0108000B):  # Bayer
        img = np.asarray(buf, dtype=np.uint8).reshape((h, w))
        code = {0x01080009: cv2.COLOR_BayerRG2BGR, 0x01080008: cv2.COLOR_BayerGR2BGR,
                0x0108000A: cv2.COLOR_BayerGB2BGR, 0x0108000B: cv2.COLOR_BayerBG2BGR}[fmt]
        return cv2.cvtColor(img, code)
    img = np.asarray(buf, dtype=np.uint8).reshape((h, w, 3))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════
# 模型后台加载
# ═══════════════════════════════════

class PipelineLoader:
    def __init__(self):
        self.yolo = None; self.hands = None; self.model = None
        self.phys = None; self.fusion = None; self.device = "cpu"
        self.ready = False
        self.error = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self):
        try:
            sys.path.insert(0, str(script_dir))
            from engine.yolo_detector import YOLODetector
            from engine.hand_detector import HandDetector
            from engine.physical_state import PhysicalStateEngine
            from engine.fusion import FusionEngine
            from engine.temporal_lstm import SOPActionGRU, FeatureExtractor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Loader] device={self.device}")

            print("[Loader] Loading YOLO...")
            self.yolo = YOLODetector(str(script_dir / "models" / "yolo_final_v1.pt"), device=self.device)
            print("[Loader] YOLO OK")

            print("[Loader] Loading MediaPipe hands...")
            self.hands = HandDetector()
            print("[Loader] Hands OK")

            print("[Loader] Loading temporal model...")
            ckpt = torch.load(str(script_dir / "training" / "runs" / "best_sequence_v5.pt"),
                            map_location=self.device)
            self.model = SOPActionGRU(
                input_size=ckpt.get("input_size", 130), hidden_size=ckpt.get("hidden_size", 256),
                num_layers=ckpt.get("num_layers", 3), num_classes=ckpt.get("num_classes", 7))
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device).eval()
            print(f"[Loader] Temporal OK (val_acc={ckpt.get('val_acc', 0):.2%})")

            self.feat_ext = FeatureExtractor(self.yolo, self.hands)
            self.phys = PhysicalStateEngine()
            self.fusion = FusionEngine(confirm_count=2, model_window=8, min_step_sec=0.4)
            self.ready = True
            print(f"[Loader] All models ready!")
        except Exception as e:
            self.error = str(e)
            with open(CRASH_LOG, "w") as f:
                f.write(f"Model load error:\n{traceback.format_exc()}")
            print(f"[Loader] ERROR: {e}")


# ═══════════════════════════════════
# 主程序
# ═══════════════════════════════════

def main():
    cam, buf, ps, W, H = init_camera()

    win = "SOP Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    loader = PipelineLoader()
    loader.start()

    running = False
    frame_idx = 0
    feature_buffer = []
    model_history = []
    fps_buf = []
    t_last = time.time()
    last_fusion = None
    error_flash = 0
    good_flash = 0

    print("\nPress 'S' to start | Q to quit")

    while True:
        # ── 取帧 ──
        frame = read_frame(cam, buf, ps)
        if frame is None:
            time.sleep(0.001)
            continue

        # FPS
        t1 = time.time(); dt = max(t1 - t_last, 0.001); t_last = t1
        fps_buf.append(1.0/dt)
        if len(fps_buf) > 30: fps_buf.pop(0)
        fps = sum(fps_buf)/len(fps_buf)

        display = frame.copy()
        Hf, Wf = display.shape[:2]

        # ── 右侧信息栏 ──
        panel = np.zeros((Hf, 280, 3), dtype=np.uint8); panel[:] = (30,30,30)
        cy = 10
        cv2.putText(panel, "SOP Detection", (10,cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cy += 35
        cv2.putText(panel, f"FPS: {fps:.1f}", (10,cy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
        cy += 28

        if not loader.ready:
            status = "Loading models..." if not loader.error else f"Model error: {loader.error[:40]}"
            cv2.putText(panel, status, (10, cy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,0), 1)
            cy += 20

        if not running:
            # ── 提示文字 ──
            bar_h = 70
            cv2.rectangle(display, (Wf//2-250, Hf//2-bar_h//2),
                         (Wf//2+250, Hf//2+bar_h//2), (0,0,0), -1)
            cv2.rectangle(display, (Wf//2-250, Hf//2-bar_h//2),
                         (Wf//2+250, Hf//2+bar_h//2), (0,200,0), 2)
            cv2.putText(display, "Press 'S' to Start", (Wf//2-220, Hf//2-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display, "SOP Detection System", (Wf//2-200, Hf//2+35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Status: IDLE", (10,cy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)
            cy += 28
            if loader.ready:
                cv2.putText(panel, "Ready - Press S", (10,cy+18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)

        else:
            if not loader.ready:
                cv2.putText(display, "Loading models...", (10, Hf-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            else:
                # ── 检测模式 ──
                try:
                    frame_idx += 1

                    # YOLO检测 (每6帧跑一次省资源)
                    detections = []
                    if frame_idx % 6 == 0:
                        detections = loader.yolo.detect(frame)

                    # 手部检测
                    hands = loader.hands.detect(frame)

                    # 盒子状态 (从YOLO结果)
                    box_state, box_conf = loader.yolo.get_box_state(detections)
                    box_bbox = loader.yolo.get_box_bbox(detections)

                    # 手物交互 & 物理状态
                    interaction = loader.hands.compute_interaction(
                        hands, detections, box_bbox, (Hf, Wf))
                    hand_near_box = any(
                        d < 0.25 for d in interaction["hand_box_dist"] if d < 999)

                    phys_result = loader.phys.update(
                        detections, box_state, box_bbox,
                        hand_near_box, len(hands) > 0,
                        hand_obj_iou=interaction["hand_obj_iou"],
                        hand_box_dist=interaction["hand_box_dist"])

                    # ── 特征提取 ──
                    feats = loader.feat_ext.extract(
                        frame, detections, hands, interaction, box_bbox, box_state)

                    feature_buffer.append(feats)
                    if len(feature_buffer) > 128:
                        feature_buffer.pop(0)

                    # ── 时序推理 ──
                    model_step = 0; model_conf = 0.0; model_top3 = []
                    if frame_idx % 6 == 0 and len(feature_buffer) >= 12:
                        buf_t = np.stack(list(feature_buffer), axis=0).astype(np.float32)
                        t = torch.from_numpy(buf_t).unsqueeze(0).to(loader.device)
                        with torch.no_grad():
                            logits = loader.model(t)[0, -1]
                            probs = torch.softmax(logits, dim=-1)
                            top3 = torch.topk(probs, 3)
                            model_step = int(top3.indices[0].item())
                            model_conf = float(top3.values[0].item())
                            model_top3 = [(int(top3.indices[i].item()),
                                          float(top3.values[i].item())) for i in range(3)]
                        model_history.append(model_step)
                        if len(model_history) > 8:
                            model_history.pop(0)
                        model_step = Counter(model_history).most_common(1)[0][0]

                    # ── 融合 ──
                    current = last_fusion.step if last_fusion else 0
                    fusion_result = loader.fusion.update(phys_result, model_step, model_conf,
                                                          model_top3, time.time())
                    if fusion_result.step != current and fusion_result.step < 6:
                        print(f"  S{current}->S{fusion_result.step} {fusion_result.step_name}")
                        if fusion_result.is_correct:
                            good_flash = 25
                    last_fusion = fusion_result

                    # ── 绘制 ──
                    if good_flash > 0:
                        good_flash -= 1
                        bar = np.zeros((50,Wf,3), dtype=np.uint8); bar[:]=(0,160,0)
                        cv2.putText(bar, f"CORRECT: {fusion_result.step_name}",
                                   (20,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                        display = np.vstack([bar, display])
                        Hf = display.shape[0]

                    if fusion_result.error_type:
                        error_flash = 20
                    if error_flash > 0:
                        error_flash -= 1
                        bar = np.zeros((50,Wf,3), dtype=np.uint8); bar[:]=(0,0,200)
                        cv2.putText(bar, f"ERROR: {fusion_result.error_type}",
                                   (20,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                        display = np.vstack([bar, display])

                    # 绘制检测框
                    for d in detections:
                        x1,y1,x2,y2 = map(int, d.bbox)
                        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(display, f"{d.cls_name} {d.confidence:.2f}",
                                   (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)

                    # 绘制手部关键点
                    for h in hands:
                        for lm in h.landmarks:
                            px, py = int(lm[0] * Wf), int(lm[1] * Hf)
                            cv2.circle(display, (px, py), 2, (0, 200, 100), -1)

                    # 面板信息
                    sc = (0,200,0) if fusion_result.is_correct else (0,0,200)
                    cv2.putText(panel, f"Step: {fusion_result.step_name}",
                               (10,cy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
                    cy += 25
                    for i in range(1,6):
                        mark = "OK" if fusion_result.step >= i else "..."
                        color = (0,200,0) if fusion_result.step >= i else (100,100,100)
                        cv2.putText(panel, f"S{i}: {mark}", (10,cy+18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cy += 20
                    cy += 8
                    cv2.putText(panel, f"Box: {'OPEN' if phys_result.box_is_open else 'CLOSED'}",
                               (10,cy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
                    cv2.putText(panel, f"Vis: {','.join(vis) if vis else 'none'}",
                               (10,cy+34), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150,150,150), 1)

                except Exception as e:
                    # 捕获所有异常，写入日志，显示错误但不崩溃
                    with open(CRASH_LOG, "w") as f:
                        f.write(f"Detection error at frame {frame_idx}:\n{traceback.format_exc()}")
                    cv2.putText(display, f"DETECTION ERROR: {e}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(f"[ERROR] {e}  (see crash_log.txt)")

        # ── 组合显示 ──
        Hf = display.shape[0]
        scale = min(960, Wf) / Wf
        d = cv2.resize(display, (int(Wf*scale), int(Hf*scale)))
        p = cv2.resize(panel, (int(280*scale), d.shape[0]))
        cv2.imshow(win, np.hstack([d, p]))

        key = cv2.waitKey(1)
        if key in (ord('q'), 27):
            break
        elif key == ord('s') and not running:
            if loader.ready:
                running = True
                loader.fusion.reset(); loader.phys.reset()
                feature_buffer.clear(); model_history.clear()
                frame_idx = 0; error_flash = 0; good_flash = 0
                print("\n=== Detection Started ===\n")
            else:
                print("  Models still loading...")
        elif key == ord('r') and running:
            running = False
            if loader.ready:
                loader.fusion.reset(); loader.phys.reset()
            feature_buffer.clear()
            print("\n=== Reset ===\n")

    # 释放
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    cv2.destroyAllWindows()
    print("\nDone")


if __name__ == "__main__":
    main()
