"""
Bridge the local SOP detector into the RuoYi SOP pages.

It does three things:
1. Runs the existing `local_run.py` pipeline.
2. Serves a lightweight MJPEG preview stream and saved step clips.
3. Pushes runtime state, step media, and real detection events to RuoYi.
"""

import argparse
import collections
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http import server
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from local_run import (
    EVENT_SEQUENCE,
    EVENT_TO_STEP_ID,
    STEP_NAMES_CN,
    SOPLocalPipeline,
    create_source,
    draw_status,
)


ROOT = Path(__file__).resolve().parent


def now_ts() -> float:
    return time.time()


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


def to_jpeg_bytes(frame: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode JPEG")
    return buf.tobytes()


def write_mp4(path: Path, frames: List[np.ndarray], fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


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
    frames: List[np.ndarray] = field(default_factory=list)
    start_index: int = 0
    event_index: int = 0
    event_time: float = 0.0
    post_frames_left: int = 0
    snapshot_url: str = ""
    clip_url: str = ""
    clip_start_ms: int = 0
    clip_end_ms: int = 0


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

    def do_GET(self):
        if self.path == "/stream.mjpg":
            self.handle_mjpeg()
            return
        if self.path == "/frames/latest.jpg":
            self.handle_latest_frame()
            return
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
                time.sleep(0.08)
        except (ConnectionResetError, BrokenPipeError):
            return

    def handle_latest_frame(self):
        frame = self.shared_state.get_frame()
        if frame is None:
            self.send_error(404, "No frame yet")
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


class ThreadingHttpServer(server.ThreadingHTTPServer):
    daemon_threads = True


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


class RuntimeBridge:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.clips_dir = self.output_dir / "clips"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.preview_state = SharedPreviewState()
        self.ruoyi = RuoyiClient(args.ruoyi_base_url)
        self.source = create_source(args)
        self.pipeline = SOPLocalPipeline(args)
        self.pipeline.reset()

        self.frame_index = 0
        self.ring_buffer: Deque[FramePacket] = collections.deque(maxlen=max(20, int(args.clip_pre_seconds * args.fps_hint)))
        self.accepted_events: List[str] = []
        self.last_alarm_key = ""
        self.last_sync_at = 0.0
        self.last_session_poll_at = 0.0
        self.pending_clips: List[PendingClip] = []
        self.step_media: Dict[int, Dict] = {}
        self.latest_frame_url = ""
        self.preview_stream_url = f"http://{args.public_host}:{args.bridge_port}/stream.mjpg"
        self.task_code = args.task_code or ""
        self.session_runtime_mode = "READY" if self.task_code else ""
        self.last_control_state = ""
        self.last_info: Dict = self.make_preview_info("bridge initialized")
        self.last_preview_publish_at = 0.0
        self.last_latest_write_at = 0.0

    def run(self):
        print(f"[Bridge] preview stream: {self.preview_stream_url}")
        print(f"[Bridge] task code: {self.task_code or '-'}")
        try:
            while True:
                if now_ts() - self.last_session_poll_at >= 0.25:
                    self.ensure_runtime_session(force_create=False)
                    self.last_session_poll_at = now_ts()

                frame, ts = self.source.read()
                if frame is None:
                    time.sleep(0.02)
                    continue

                self.frame_index += 1
                det_summary, info, box_bbox, box_state, hands = self.pipeline.process_frame(frame, ts)
                self.last_info = info
                overlay = draw_status(frame.copy(), det_summary, hands, box_bbox, box_state, info, local_fps=info.get("fps", 0.0))

                self.handle_preview_frame(overlay)
                self.handle_ring_buffer(overlay, ts)
                self.handle_step_events(info, overlay, ts)
                self.update_pending_clips(overlay)

                if now_ts() - self.last_sync_at >= self.args.sync_interval:
                    self.push_runtime(info)
        except KeyboardInterrupt:
            print("[Bridge] stopped by user")
        finally:
            self.push_runtime(self.make_preview_info("bridge stopped"), force=True)
            self.source.close()

    def handle_preview_frame(self, frame: np.ndarray) -> None:
        now = now_ts()
        publish_interval = 1.0 / max(self.args.preview_fps, 0.1)
        if now - self.last_preview_publish_at < publish_interval and self.preview_state.get_frame() is not None:
            return

        jpeg_bytes = to_jpeg_bytes(frame, quality=self.args.preview_quality)
        self.preview_state.set_frame(jpeg_bytes)
        self.last_preview_publish_at = now

        if now - self.last_latest_write_at >= self.args.latest_write_interval:
            latest_path = self.frames_dir / "latest.jpg"
            latest_path.write_bytes(jpeg_bytes)
            self.latest_frame_url = f"http://{self.args.public_host}:{self.args.bridge_port}/frames/latest.jpg"
            self.last_latest_write_at = now

    def handle_ring_buffer(self, frame: np.ndarray, ts: float) -> None:
        self.ring_buffer.append(FramePacket(frame=frame.copy(), timestamp=ts, frame_index=self.frame_index))

    def handle_step_events(self, info: Dict, frame: np.ndarray, ts: float) -> None:
        current_accepted = list(info.get("accepted", []))
        if len(current_accepted) > len(self.accepted_events):
            for event_code in current_accepted[len(self.accepted_events):]:
                self.accepted_events.append(event_code)
                self.register_event(event_code, frame, ts, info, is_alarm=False)

        alarm = info.get("alarm")
        if alarm:
            alarm_key = f"{alarm.get('type')}::{alarm.get('message')}"
            if alarm_key != self.last_alarm_key:
                self.last_alarm_key = alarm_key
                self.register_event(alarm.get("type", "ALARM").lower(), frame, ts, info, is_alarm=True)
        else:
            self.last_alarm_key = ""

    def register_event(self, event_code: str, frame: np.ndarray, ts: float, info: Dict, is_alarm: bool) -> None:
        if not self.task_code:
            return
        step_no = EVENT_TO_STEP_ID.get(event_code, min(len(self.accepted_events), 5))
        snapshot_name = f"{self.task_code}_s{step_no}_{int(ts * 1000)}.jpg"
        snapshot_path = self.frames_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), frame)
        snapshot_url = f"http://{self.args.public_host}:{self.args.bridge_port}/frames/{snapshot_name}"

        pre_frames = [packet.frame.copy() for packet in self.ring_buffer]
        start_index = self.ring_buffer[0].frame_index if self.ring_buffer else self.frame_index
        pending = PendingClip(
            step_no=step_no,
            event_code=event_code,
            pre_frames=pre_frames,
            start_index=start_index,
            event_index=self.frame_index,
            event_time=ts,
            post_frames_left=max(1, int(self.args.clip_post_seconds * self.args.fps_hint)),
            snapshot_url=snapshot_url,
        )
        self.pending_clips.append(pending)

        media = self.step_media.setdefault(step_no, {})
        media["snapshotUrl"] = snapshot_url
        media["judgeResult"] = "FAIL" if is_alarm else "PASS"
        media["judgeMessage"] = alarm_message(info) if is_alarm else f"{event_code} detected"

        payload = self.build_runtime_payload(info)
        payload["event"] = {
            "requestId": f"{self.task_code}_{step_no}_{int(ts * 1000)}",
            "taskCode": self.task_code,
            "eventCode": event_code,
            "eventName": event_code,
            "confidence": round(runtime_confidence(info), 4),
            "imageUrl": snapshot_url,
        }
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
            frames = clip.pre_frames + clip.frames
            write_mp4(clip_path, frames, self.args.fps_hint)
            clip.clip_url = f"http://{self.args.public_host}:{self.args.bridge_port}/clips/{clip_name}"
            clip.clip_start_ms = max(0, int((clip.event_index - clip.start_index) / max(self.args.fps_hint, 1.0) * 1000) - int(self.args.clip_pre_seconds * 1000))
            clip.clip_end_ms = clip.clip_start_ms + int(len(frames) / max(self.args.fps_hint, 1.0) * 1000)

            media = self.step_media.setdefault(clip.step_no, {})
            media["snapshotUrl"] = clip.snapshot_url
            media["clipUrl"] = clip.clip_url
            media["clipStartMs"] = clip.clip_start_ms
            media["clipEndMs"] = clip.clip_end_ms

    def push_runtime(self, info: Dict, force: bool = False) -> None:
        if not force and not self.task_code:
            return
        payload = self.build_runtime_payload(info)
        self.ruoyi.sync(payload)
        self.last_sync_at = now_ts()

    def build_runtime_payload(self, info: Dict) -> Dict:
        accepted_count = len(self.accepted_events)
        mode = self.effective_runtime_mode(info)
        task_status = task_status_from_mode(mode)
        current_step_no = min(max(accepted_count + 1, 1), 5)
        steps = []

        for step_no in range(1, 6):
            media = self.step_media.get(step_no, {})
            if step_no <= accepted_count:
                step_status = "PASSED"
            elif task_status == "FAILED" and step_no == current_step_no:
                step_status = "FAILED"
            elif task_status in ("RUNNING",) and step_no == current_step_no:
                step_status = "RUNNING"
            else:
                step_status = "PENDING"

            steps.append({
                "stepNo": step_no,
                "stepName": STEP_NAMES_CN.get(step_no, f"S{step_no}"),
                "expectedEvent": EVENT_SEQUENCE[step_no - 1] if step_no - 1 < len(EVENT_SEQUENCE) else "",
                "stepStatus": step_status,
                "snapshotUrl": media.get("snapshotUrl"),
                "clipUrl": media.get("clipUrl"),
                "clipStartMs": media.get("clipStartMs"),
                "clipEndMs": media.get("clipEndMs"),
                "judgeResult": media.get("judgeResult"),
                "judgeMessage": media.get("judgeMessage"),
            })

        runtime_message = self.effective_runtime_message(mode, info)

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
            "runtimeMessage": runtime_message,
            "runtimeFps": round(float(info.get("fps", 0.0)), 2),
            "steps": steps,
        }

    def make_preview_info(self, message: str) -> Dict:
        next_index = min(len(self.accepted_events) + 1, 5)
        return {
            "mode": "PREVIEW",
            "fps": 0.0,
            "expected_event": EVENT_SEQUENCE[next_index - 1] if next_index - 1 < len(EVENT_SEQUENCE) else EVENT_SEQUENCE[-1],
            "accepted": list(self.accepted_events),
            "alarm": None,
            "runtime_message": message,
        }

    def effective_runtime_mode(self, info: Dict) -> str:
        desired = (self.session_runtime_mode or "").upper()
        if desired in ("READY", "STOPPED"):
            return desired
        return (info.get("mode", "PREVIEW") or "PREVIEW").upper()

    def effective_runtime_message(self, mode: str, info: Dict) -> str:
        if mode == "READY":
            return "未开始，点击开始检测后进入识别"
        if mode == "STOPPED":
            return "已停止，当前仅保留实时预览"
        runtime_message = info.get("runtime_message")
        if runtime_message:
            return runtime_message
        if info.get("alarm"):
            return alarm_message(info)
        return f"{mode} | expected={info.get('expected_event', '-')}"

    def ensure_runtime_session(self, force_create: bool = False) -> None:
        if self.args.task_code:
            if self.task_code != self.args.task_code:
                self.switch_task(self.args.task_code, "READY")
            return

        session = self.ruoyi.get_current_runtime(self.args.product_id, self.args.sop_id)
        task = ((session or {}).get("data") or {}).get("task")
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
            self.switch_task(new_task_code, desired_mode)
            return
        if new_task_code:
            self.task_code = new_task_code
            self.session_runtime_mode = desired_mode
            self.apply_control_state(desired_mode)

    def desired_session_mode(self, task: Dict) -> str:
        runtime_mode = str(task.get("runtimeMode") or "").upper()
        task_status = str(task.get("taskStatus") or "").upper()
        if runtime_mode in ("READY", "STOPPED", "ARMED", "RUNNING", "COMPLETE", "ERROR"):
            return runtime_mode
        if task_status == "STOPPED":
            return "STOPPED"
        if task_status == "RUNNING":
            return "ARMED"
        return "READY"

    def switch_task(self, new_task_code: str, desired_mode: str) -> None:
        self.task_code = new_task_code
        self.session_runtime_mode = (desired_mode or "READY").upper()
        self.last_control_state = ""
        self.accepted_events = []
        self.last_alarm_key = ""
        self.pending_clips = []
        self.step_media = {}
        self.ring_buffer.clear()
        self.pipeline.reset()
        self.last_info = self.make_preview_info(f"session ready: {new_task_code}")
        self.apply_control_state(self.session_runtime_mode, force=True)

    def apply_control_state(self, desired_mode: str, force: bool = False) -> None:
        desired_mode = (desired_mode or "READY").upper()
        if desired_mode in ("READY", "STOPPED", "PREVIEW"):
            target_state = "STOPPED" if desired_mode == "STOPPED" else "READY"
            if force or self.last_control_state != target_state:
                self.pipeline.reset()
                self.last_control_state = target_state
                self.last_info = self.make_preview_info(
                    "stopped" if desired_mode == "STOPPED" else "ready"
                )
            return

        if desired_mode in ("ARMED", "RUNNING"):
            if force or self.last_control_state != "RUNNING":
                self.pipeline.reset()
                self.pipeline.start()
                self.last_control_state = "RUNNING"


def runtime_confidence(info: Dict) -> float:
    probs = info.get("latest_step_probs") or []
    return max(probs) if probs else 0.95


def alarm_message(info: Dict) -> str:
    alarm = info.get("alarm") or {}
    return f"{alarm.get('type', 'ALARM')}: {alarm.get('message', '')}".strip()


def start_http_server(shared_state: SharedPreviewState, base_dir: Path, host: str, port: int) -> ThreadingHttpServer:
    PreviewHttpHandler.shared_state = shared_state
    PreviewHttpHandler.base_dir = base_dir
    httpd = ThreadingHttpServer((host, port), PreviewHttpHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge SOP detector results into RuoYi")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", choices=["hik", "webcam"], default="webcam")
    src.add_argument("--video")
    parser.add_argument("--webcam-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--yolo")
    parser.add_argument("--model")
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-display", action="store_true")

    parser.add_argument("--ruoyi-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=int, default=18081)
    parser.add_argument("--output-dir", default=str(ROOT / "runtime_outputs"))
    parser.add_argument("--task-code", default="")
    parser.add_argument("--product-id", type=int, required=True)
    parser.add_argument("--sop-id", type=int, required=True)
    parser.add_argument("--station-code", default="ST01")
    parser.add_argument("--camera-code", default="CAM01")
    parser.add_argument("--operator-name", default="vision")
    parser.add_argument("--sync-interval", type=float, default=2.0)
    parser.add_argument("--fps-hint", type=float, default=20.0)
    parser.add_argument("--clip-pre-seconds", type=float, default=2.0)
    parser.add_argument("--clip-post-seconds", type=float, default=2.0)
    parser.add_argument("--preview-fps", type=float, default=5.0)
    parser.add_argument("--preview-quality", type=int, default=80)
    parser.add_argument("--latest-write-interval", type=float, default=1.0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    httpd = start_http_server(SharedPreviewState(), Path(args.output_dir).resolve(), args.bridge_host, args.bridge_port)
    try:
        bridge = RuntimeBridge(args)
        httpd.shutdown()
        httpd.server_close()
        httpd = start_http_server(bridge.preview_state, bridge.output_dir, args.bridge_host, args.bridge_port)
        bridge.run()
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
