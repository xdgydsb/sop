"""
Bridge: Hik USB camera -> WebSocket inference server -> RuoYi runtime page.

Preferred when the local laptop is CPU-only and the server provides faster SOP inference.
"""

import argparse
import collections
import json
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http import server
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from local_client.hik_stream_runtime import (
    MV_FRAME_OUT,
    MV_OK,
    byref,
    close_hik_camera,
    convert_frame_to_bgr,
    draw_status,
    memset,
    open_hik_camera,
    sizeof,
    ws_connect,
)


ROOT = Path(__file__).resolve().parent


def to_jpeg_bytes(frame: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode jpeg")
    return buf.tobytes()


def write_mp4(path: Path, frames: List[np.ndarray], fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def task_status_from_mode(mode: str) -> str:
    if mode == "STOPPED":
        return "STOPPED"
    if mode == "COMPLETE":
        return "PASSED"
    if mode == "ERROR":
        return "FAILED"
    if mode in ("RUNNING", "ARMED"):
        return "RUNNING"
    return "CREATED"


class SharedPreviewState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self.lock:
            self.latest_jpeg = jpeg_bytes

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg


class PreviewHttpHandler(server.SimpleHTTPRequestHandler):
    shared_state: SharedPreviewState = None
    base_dir: Path = None
    frame_interval: float = 0.05

    def do_GET(self):
        request_path = self.path.split("?", 1)[0]
        if request_path == "/stream.mjpg":
            return self.handle_mjpeg()
        if request_path == "/frames/latest.jpg":
            return self.handle_latest_frame()
        return super().do_GET()

    def handle_mjpeg(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame = self.shared_state.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(self.frame_interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def handle_latest_frame(self):
        frame = self.shared_state.get_frame()
        if frame is None:
            self.send_error(404, "no frame")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def translate_path(self, path):
        clean = path.lstrip("/")
        safe_path = (self.base_dir / clean).resolve()
        if self.base_dir not in safe_path.parents and safe_path != self.base_dir:
            return str(self.base_dir)
        return str(safe_path)

    def log_message(self, format, *args):
        return


class RuoyiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.sync_url = self.base_url + "/sop/runtime/sync"
        self.current_url = self.base_url + "/sop/runtime/current"
        self.start_url = self.base_url + "/sop/runtime/session/start"
        self.reset_url = self.base_url + "/sop/runtime/session/reset"
        self.stop_url = self.base_url + "/sop/runtime/session/stop"

    def _request_json(self, url: str, payload: Optional[Dict] = None, method: str = "GET") -> Optional[Dict]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None

    def sync(self, payload: Dict) -> Optional[Dict]:
        try:
            return self._request_json(self.sync_url, payload, method="POST")
        except urllib.error.URLError as exc:
            print(f"[Bridge] sync failed: {exc}")
            return None

    def get_current_runtime(self, product_id: int, sop_id: int) -> Optional[Dict]:
        query = urllib.parse.urlencode({"productId": product_id, "sopId": sop_id})
        try:
            return self._request_json(f"{self.current_url}?{query}")
        except urllib.error.URLError as exc:
            print(f"[Bridge] get current runtime failed: {exc}")
            return None

    def start_session(self, payload: Dict) -> Optional[Dict]:
        try:
            return self._request_json(self.start_url, payload, method="POST")
        except urllib.error.URLError as exc:
            print(f"[Bridge] start session failed: {exc}")
            return None

    def reset_session(self, payload: Dict) -> Optional[Dict]:
        try:
            return self._request_json(self.reset_url, payload, method="POST")
        except urllib.error.URLError as exc:
            print(f"[Bridge] reset session failed: {exc}")
            return None

    def stop_session(self, payload: Dict) -> Optional[Dict]:
        try:
            return self._request_json(self.stop_url, payload, method="POST")
        except urllib.error.URLError as exc:
            print(f"[Bridge] stop session failed: {exc}")
            return None


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    frame_index: int


@dataclass
class PendingClip:
    step_no: int
    event_code: str
    pre_frames: List[np.ndarray]
    start_index: int
    event_index: int
    event_time: float
    post_frames_left: int
    snapshot_url: str = ""
    frames: List[np.ndarray] = field(default_factory=list)


class HikWsRuoyiBridge:
    STEP_EVENT_BY_INDEX = {
        1: "box_opened",
        2: "earphone_in_box",
        3: "charger_in_box",
        4: "green_bag_in_box",
        5: "box_closed",
    }

    STEP_NAME_BY_INDEX = {
        1: "1. 打开盒子",
        2: "2. 放入耳机",
        3: "3. 放入充电头",
        4: "4. 放入绿色小袋",
        5: "5. 关闭盒子",
    }

    def __init__(self, args):
        self.args = args
        self.preview_state = SharedPreviewState()
        self.ruoyi = RuoyiClient(args.ruoyi_base_url)
        self.output_dir = Path(args.output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.clips_dir = self.output_dir / "clips"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.preview_stream_url = f"http://{args.public_host}:{args.bridge_port}/stream.mjpg"
        self.latest_frame_url = ""
        self.frame_index = 0
        self.ring_buffer: Deque[FramePacket] = collections.deque(
            maxlen=max(20, int(args.clip_pre_seconds * args.fps))
        )
        self.accepted_events: List[str] = []
        self.step_media: Dict[int, Dict] = {}
        self.pending_clips: List[PendingClip] = []
        self.last_alarm_key = ""
        self.last_sync_at = 0.0
        self.last_session_poll_at = 0.0
        self.task_code = args.task_code or ""
        self.session_runtime_mode = "READY" if self.task_code else ""
        self.last_control_state = ""
        self.last_server_info: Dict = self.make_preview_info("bridge initialized")
        self.last_preview_publish_at = 0.0
        self.last_latest_write_at = 0.0

    def clear_detection_state(self) -> None:
        self.accepted_events = []
        self.step_media = {}
        self.pending_clips = []
        self.ring_buffer.clear()
        self.last_alarm_key = ""
        self.last_server_info = self.make_preview_info(
            f"session ready: {self.task_code}" if self.task_code else "session ready"
        )

    def run(self):
        cam, _, _ = open_hik_camera(target_w=self.args.width, target_h=self.args.height)
        if cam is None:
            raise RuntimeError("failed to open hik camera")

        server_url = f"ws://{self.args.server}:{self.args.port}"
        try:
            while True:
                print(f"[Bridge] connect {server_url}")
                try:
                    with ws_connect(server_url, max_size=5 * 1024 * 1024) as ws:
                        self.last_control_state = ""
                        self.ensure_runtime_session(ws, force_create=False)

                        server_info = None
                        send_interval = 1.0 / max(self.args.fps, 0.1)
                        last_send = 0.0
                        fps_times = []
                        send_times = []
                        local_fps = 0.0
                        send_fps = 0.0
                        sequence_active = True

                        while True:
                            now = time.time()
                            if now - self.last_session_poll_at >= 0.25:
                                self.ensure_runtime_session(ws, force_create=False)
                                self.last_session_poll_at = now

                            st_frame = MV_FRAME_OUT()
                            memset(byref(st_frame), 0, sizeof(MV_FRAME_OUT))
                            ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
                            if ret != MV_OK or st_frame.pBufAddr is None:
                                time.sleep(0.005)
                                continue

                            try:
                                frame = convert_frame_to_bgr(st_frame)
                            finally:
                                cam.MV_CC_FreeImageBuffer(st_frame)

                            now = time.time()
                            self.frame_index += 1
                            fps_times.append(now)
                            if len(fps_times) > 30:
                                fps_times = fps_times[-30:]
                            if len(fps_times) >= 2:
                                local_fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-3)

                            if now - last_send >= send_interval:
                                last_send = now
                                send_frame = cv2.resize(frame, (self.args.width, self.args.height))
                                jpeg_bytes = to_jpeg_bytes(send_frame, quality=self.args.quality)
                                header = struct.pack("<Id", self.frame_index, now)
                                ws.send(header + jpeg_bytes)
                                send_times.append(now)
                                if len(send_times) > 30:
                                    send_times = send_times[-30:]
                                if len(send_times) >= 2:
                                    send_fps = (len(send_times) - 1) / max(send_times[-1] - send_times[0], 1e-3)

                            try:
                                response = ws.recv(timeout=0.001)
                                if response:
                                    msg = json.loads(response)
                                    if "control_ack" not in msg:
                                        server_info = msg
                                        self.last_server_info = server_info
                                        mode = str(server_info.get("mode", "PREVIEW")).upper()
                                        sequence_active = mode in ("ARMED", "RUNNING", "COMPLETE", "ERROR")
                            except TimeoutError:
                                pass
                            except Exception:
                                pass

                            display = cv2.resize(frame, (self.args.width, self.args.height))
                            display = draw_status(
                                display,
                                local_fps,
                                send_fps,
                                server_info,
                                self.args.width,
                                self.args.height,
                                True,
                                sequence_active,
                            )
                            self.handle_preview_frame(display)
                            self.handle_ring_buffer(display, now)

                            if server_info:
                                self.handle_server_events(server_info, display, now)
                                self.update_pending_clips(display)
                            else:
                                self.last_server_info = self.make_preview_info("waiting for inference server")

                            if time.time() - self.last_sync_at >= self.args.sync_interval:
                                self.ensure_runtime_session(ws, force_create=False)
                                self.push_runtime(self.last_server_info)
                except Exception as exc:
                    self.last_server_info = self.make_preview_info(f"inference reconnecting: {exc}")
                    self.safe_sync(self.last_server_info)
                    print(f"[Bridge] websocket disconnected: {exc}")
                    time.sleep(2)
        finally:
            self.safe_sync(self.make_preview_info("bridge stopped"))
            close_hik_camera(cam)

    def handle_preview_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        publish_interval = 1.0 / max(self.args.preview_fps, 0.1)
        jpeg_bytes = None

        if now - self.last_preview_publish_at >= publish_interval or self.preview_state.get_frame() is None:
            jpeg_bytes = to_jpeg_bytes(frame, quality=self.args.preview_quality)
            self.preview_state.set_frame(jpeg_bytes)
            self.last_preview_publish_at = now

        if now - self.last_latest_write_at >= self.args.latest_write_interval:
            if jpeg_bytes is None:
                jpeg_bytes = to_jpeg_bytes(frame, quality=self.args.preview_quality)
            latest = self.frames_dir / "latest.jpg"
            latest.write_bytes(jpeg_bytes)
            self.latest_frame_url = f"http://{self.args.public_host}:{self.args.bridge_port}/frames/latest.jpg"
            self.last_latest_write_at = now

    def handle_ring_buffer(self, frame: np.ndarray, ts: float) -> None:
        self.ring_buffer.append(FramePacket(frame.copy(), ts, self.frame_index))

    def handle_server_events(self, info: Dict, frame: np.ndarray, ts: float) -> None:
        accepted = list(info.get("accepted_events") or [])
        previous = list(self.accepted_events)
        new_events = [event_code for event_code in accepted if event_code not in previous]
        if new_events:
            self.accepted_events = accepted
            current_event = info.get("event_name")
            if current_event in new_events:
                self.capture_step_event(current_event, frame, ts, info, is_alarm=False)
            elif len(new_events) == 1:
                self.capture_step_event(new_events[0], frame, ts, info, is_alarm=False)
            else:
                # Do not synthesize several step clips from one frame. If the
                # bridge missed old events, keep their PASS state but avoid fake
                # identical videos for S1-S5.
                print(f"[Bridge] skipped synthetic clips for batched events: {new_events}")
        else:
            self.accepted_events = accepted

        alarm = info.get("alarm")
        if alarm:
            alarm_key = f"{alarm.get('type')}::{alarm.get('message')}"
            if alarm_key != self.last_alarm_key:
                self.last_alarm_key = alarm_key
                self.capture_step_event(alarm.get("type", "alarm").lower(), frame, ts, info, is_alarm=True)
        else:
            self.last_alarm_key = ""

    def capture_step_event(self, event_code: str, frame: np.ndarray, ts: float, info: Dict, is_alarm: bool) -> None:
        if not self.task_code:
            return
        step_no = self.event_to_step(event_code, info)
        media = self.step_media.setdefault(step_no, {})
        if not is_alarm and media.get("snapshotUrl"):
            return
        snapshot_name = f"{self.task_code}_s{step_no}_{int(ts * 1000)}.jpg"
        snapshot_path = self.frames_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), frame)
        snapshot_url = f"http://{self.args.public_host}:{self.args.bridge_port}/frames/{snapshot_name}"

        pending = PendingClip(
            step_no=step_no,
            event_code=event_code,
            pre_frames=[item.frame.copy() for item in self.ring_buffer],
            start_index=self.ring_buffer[0].frame_index if self.ring_buffer else self.frame_index,
            event_index=self.frame_index,
            event_time=ts,
            post_frames_left=max(1, int(self.args.clip_post_seconds * self.args.fps)),
            snapshot_url=snapshot_url,
        )
        self.pending_clips.append(pending)

        keep_passed_step = is_alarm and media.get("judgeResult") == "PASS"
        if not keep_passed_step:
            media["snapshotUrl"] = snapshot_url
            media["judgeResult"] = "FAIL" if is_alarm else "PASS"
            media["judgeMessage"] = self.alarm_message(info) if is_alarm else f"{event_code} detected"

        payload = self.build_runtime_payload(info)
        self.ruoyi.sync(payload)

    def update_pending_clips(self, frame: np.ndarray) -> None:
        finished = []
        for clip in self.pending_clips:
            clip.frames.append(frame.copy())
            clip.post_frames_left -= 1
            if clip.post_frames_left <= 0:
                finished.append(clip)

        for clip in finished:
            self.pending_clips.remove(clip)
            clip_name = f"{self.task_code}_s{clip.step_no}_{int(clip.event_time * 1000)}.mp4"
            clip_path = self.clips_dir / clip_name
            all_frames = clip.pre_frames + clip.frames
            write_mp4(clip_path, all_frames, self.args.fps)
            clip_url = f"http://{self.args.public_host}:{self.args.bridge_port}/clips/{clip_name}"
            clip_start_ms = max(
                0,
                int((clip.event_index - clip.start_index) / max(self.args.fps, 1.0) * 1000)
                - int(self.args.clip_pre_seconds * 1000),
            )
            clip_end_ms = clip_start_ms + int(len(all_frames) / max(self.args.fps, 1.0) * 1000)

            media = self.step_media.setdefault(clip.step_no, {})
            media["clipUrl"] = clip_url
            media["clipStartMs"] = clip_start_ms
            media["clipEndMs"] = clip_end_ms
            self.safe_sync(self.last_server_info)

    def push_runtime(self, info: Dict) -> None:
        self.safe_sync(info)
        self.last_sync_at = time.time()

    def safe_sync(self, info: Dict) -> None:
        if not self.task_code:
            return
        if self.session_runtime_mode in ("COMPLETE", "ERROR", "PASSED", "FAILED", "FINISHED"):
            return
        try:
            self.ruoyi.sync(self.build_runtime_payload(info))
        except Exception as exc:
            print(f"[Bridge] safe sync failed: {exc}")

    def make_preview_info(self, message: str) -> Dict:
        next_index = min(len(self.accepted_events) + 1, 5)
        return {
            "mode": "PREVIEW",
            "expected_event": self.STEP_EVENT_BY_INDEX[next_index],
            "accepted_events": list(self.accepted_events),
            "server_fps": 0.0,
            "runtime_message": message,
            "alarm": None,
        }

    def build_runtime_payload(self, info: Dict) -> Dict:
        mode = self.effective_runtime_mode(info)
        task_status = task_status_from_mode(mode)
        accepted_count = len(self.accepted_events)
        current_step_no = 5 if mode == "COMPLETE" else min(max(accepted_count + 1, 1), 5)

        steps = []
        for step_no in range(1, 6):
            media = self.step_media.get(step_no, {})
            if step_no <= accepted_count:
                step_status = "PASSED"
            elif task_status == "FAILED" and step_no == current_step_no:
                step_status = "FAILED"
            elif task_status == "RUNNING" and step_no == current_step_no:
                step_status = "RUNNING"
            else:
                step_status = "PENDING"
            show_judge = step_status in ("PASSED", "FAILED")
            steps.append({
                "stepNo": step_no,
                "stepName": self.step_name(step_no),
                "expectedEvent": self.STEP_EVENT_BY_INDEX[step_no],
                "stepStatus": step_status,
                "snapshotUrl": media.get("snapshotUrl"),
                "clipUrl": media.get("clipUrl"),
                "clipStartMs": media.get("clipStartMs"),
                "clipEndMs": media.get("clipEndMs"),
                "judgeResult": media.get("judgeResult") if show_judge else None,
                "judgeMessage": media.get("judgeMessage") if show_judge else None,
            })

        return {
            "taskCode": self.task_code,
            "productId": self.args.product_id,
            "sopId": self.args.sop_id,
            "stationCode": self.args.station_code,
            "cameraCode": self.args.camera_code,
            "currentStepNo": current_step_no,
            "taskStatus": task_status,
            "operatorName": self.args.operator_name,
            "previewStreamUrl": self.preview_stream_url,
            "latestFrameUrl": self.latest_frame_url,
            "runtimeMode": mode,
            "runtimeMessage": self.effective_runtime_message(mode, info),
            "runtimeFps": round(float(info.get("server_fps", 0.0)), 2),
            "steps": steps,
        }

    def step_name(self, step_no: int) -> str:
        return self.STEP_NAME_BY_INDEX.get(step_no, f"S{step_no}")

    def effective_runtime_mode(self, info: Dict) -> str:
        desired = (self.session_runtime_mode or "").upper()
        if desired in ("COMPLETE", "ERROR"):
            return desired
        if desired in ("READY", "STOPPED"):
            return desired
        mode = (info.get("mode", "PREVIEW") or "PREVIEW").upper()
        if desired in ("ARMED", "RUNNING") and mode in ("PREVIEW", "READY"):
            # The inference server needs a few frames after START before it reports
            # ARMED/RUNNING. Keep the RuoYi task armed during that handoff instead
            # of syncing an older preview frame back as CREATED.
            return desired
        return mode

    def effective_runtime_message(self, mode: str, info: Dict) -> str:
        if mode == "READY":
            return "未开始，点击开始检测后进入识别"
        if mode == "STOPPED":
            return "已停止，当前仅保留实时预览"
        return info.get("runtime_message") or (
            self.alarm_message(info) if info.get("alarm") else f"{mode} | expected={info.get('expected_event', '-')}"
        )

    def event_to_step(self, event_code: str, info: Dict) -> int:
        for step_no, step_event in self.STEP_EVENT_BY_INDEX.items():
            if event_code == step_event:
                return step_no
        current = int(info.get("current_step_id", 1) or 1)
        return min(max(current, 1), 5)

    def alarm_message(self, info: Dict) -> str:
        alarm = info.get("alarm") or {}
        return f"{alarm.get('type', 'ALARM')}: {alarm.get('message', '')}".strip()

    def ensure_runtime_session(self, ws, force_create: bool = False) -> None:
        session = self.ruoyi.get_current_runtime(self.args.product_id, self.args.sop_id)
        task = ((session or {}).get("data") or {}).get("task")
        if self.args.task_code and task and task.get("taskCode") != self.args.task_code:
            task = None

        if self.args.task_code and task is None:
            if self.task_code != self.args.task_code:
                self.switch_task(ws, self.args.task_code, "READY")
            return

        if task is None and force_create:
            session = self.ruoyi.start_session({
                "productId": self.args.product_id,
                "sopId": self.args.sop_id,
                "stationCode": self.args.station_code,
                "cameraCode": self.args.camera_code,
                "operatorName": self.args.operator_name,
            })
            task = ((session or {}).get("data") or {}).get("task")

        if task is None:
            return

        new_task_code = task.get("taskCode") or ""
        desired_mode = self.desired_session_mode(task)
        if new_task_code and new_task_code != self.task_code:
            self.switch_task(ws, new_task_code, desired_mode)
            return
        if new_task_code:
            self.session_runtime_mode = desired_mode
            self.apply_control_state(ws, desired_mode)

    def desired_session_mode(self, task: Dict) -> str:
        runtime_mode = str(task.get("runtimeMode") or "").upper()
        task_status = str(task.get("taskStatus") or "").upper()
        if runtime_mode in ("READY", "STOPPED", "ARMED", "RUNNING", "COMPLETE", "ERROR"):
            return runtime_mode
        if task_status == "PASSED":
            return "COMPLETE"
        if task_status == "FAILED":
            return "ERROR"
        if task_status == "STOPPED":
            return "STOPPED"
        if task_status == "RUNNING":
            return "ARMED"
        return "READY"

    def switch_task(self, ws, new_task_code: str, desired_mode: str) -> None:
        self.task_code = new_task_code
        self.session_runtime_mode = (desired_mode or "READY").upper()
        self.last_control_state = ""
        self.clear_detection_state()
        if self.session_runtime_mode in ("COMPLETE", "ERROR", "PASSED", "FAILED", "FINISHED"):
            return
        try:
            ws.send("reset")
        except Exception as exc:
            print(f"[Bridge] failed to reset inference session: {exc}")
        self.apply_control_state(ws, self.session_runtime_mode, force=True)

    def apply_control_state(self, ws, desired_mode: str, force: bool = False) -> None:
        desired_mode = (desired_mode or "READY").upper()
        if desired_mode in ("READY", "STOPPED", "PREVIEW"):
            target_state = "STOPPED" if desired_mode == "STOPPED" else "READY"
            if force or self.last_control_state != target_state:
                self.clear_detection_state()
                try:
                    ws.send("stop" if desired_mode == "STOPPED" else "reset")
                except Exception as exc:
                    print(f"[Bridge] failed to apply {desired_mode}: {exc}")
                self.last_control_state = target_state
                self.last_server_info = self.make_preview_info(
                    "stopped" if desired_mode == "STOPPED" else "ready"
                )
            return

        if desired_mode in ("ARMED", "RUNNING"):
            if force or self.last_control_state != "RUNNING":
                self.clear_detection_state()
                try:
                    ws.send("reset")
                    time.sleep(0.05)
                    ws.send("start")
                except Exception as exc:
                    print(f"[Bridge] failed to start inference session: {exc}")
                self.last_control_state = "RUNNING"


def start_http_server(shared_state: SharedPreviewState, base_dir: Path, host: str, port: int, frame_interval: float):
    PreviewHttpHandler.shared_state = shared_state
    PreviewHttpHandler.base_dir = base_dir
    PreviewHttpHandler.frame_interval = max(frame_interval, 0.01)
    httpd = server.ThreadingHTTPServer((host, port), PreviewHttpHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def build_parser():
    parser = argparse.ArgumentParser(description="Hik camera -> WS inference -> RuoYi bridge")
    parser.add_argument("--server", default="192.168.31.19")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--preview-fps", type=float, default=20.0)
    parser.add_argument("--preview-quality", type=int, default=82)
    parser.add_argument("--latest-write-interval", type=float, default=0.3)

    parser.add_argument("--ruoyi-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=int, default=18081)
    parser.add_argument("--output-dir", default=str(ROOT / "runtime_outputs_ws"))
    parser.add_argument("--task-code", default="")
    parser.add_argument("--product-id", type=int, required=True)
    parser.add_argument("--sop-id", type=int, required=True)
    parser.add_argument("--station-code", default="STATION-01")
    parser.add_argument("--camera-code", default="MV-CS050-10UC")
    parser.add_argument("--operator-name", default="vision")
    parser.add_argument("--sync-interval", type=float, default=0.5)
    parser.add_argument("--clip-pre-seconds", type=float, default=2.0)
    parser.add_argument("--clip-post-seconds", type=float, default=2.0)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = HikWsRuoyiBridge(args)
    httpd = start_http_server(
        bridge.preview_state,
        output_dir,
        args.bridge_host,
        args.bridge_port,
        1.0 / max(args.preview_fps, 0.1),
    )
    try:
        bridge.run()
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
