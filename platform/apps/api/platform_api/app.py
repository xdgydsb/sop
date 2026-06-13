"""Flask application for the first browser-visible platform slice."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from .legacy_runtime import (
    LegacyRuntimeAdapter,
    LegacyRuntimeConfig,
    RuntimeUnavailable,
)
from .store import CatalogStore


def _probe(url: str, timeout: float = 1.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            online = 200 <= response.status < 400
            return {
                "online": online,
                "latencyMs": round((time.perf_counter() - started) * 1000, 1),
                "message": f"HTTP {response.status}",
            }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "online": False,
            "latencyMs": None,
            "message": str(exc),
        }


def create_app(
    runtime_adapter: LegacyRuntimeAdapter | None = None,
    catalog_store: CatalogStore | None = None,
) -> Flask:
    app_root = Path(__file__).resolve().parents[1]
    web_root = app_root.parent / "web"
    static_root = web_root / "static"
    app = Flask(
        __name__,
        static_folder=str(static_root),
        static_url_path="/static",
    )

    runtime = runtime_adapter or LegacyRuntimeAdapter(
        LegacyRuntimeConfig(
            base_url=os.getenv("PLATFORM_RUOYI_URL", "http://127.0.0.1:8080"),
            product_id=int(os.getenv("PLATFORM_PRODUCT_ID", "100")),
            sop_id=int(os.getenv("PLATFORM_SOP_ID", "100")),
            station_code=os.getenv("PLATFORM_STATION_CODE", "STATION-01"),
            camera_code=os.getenv("PLATFORM_CAMERA_CODE", "MV-CS050-10UC"),
        )
    )
    store = catalog_store or CatalogStore(
        Path(
            os.getenv(
                "PLATFORM_CATALOG_PATH",
                str(app_root.parents[2] / "runtime_logs" / "platform" / "platform.db"),
            )
        )
    )
    camera_url = os.getenv(
        "PLATFORM_CAMERA_URL", "http://127.0.0.1:18081/stream.mjpg"
    )
    camera_health_url = os.getenv(
        "PLATFORM_CAMERA_HEALTH_URL",
        "http://127.0.0.1:18081/frames/latest.jpg",
    )

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/platform/catalog")
    def get_catalog():
        return jsonify({"ok": True, "data": store.get()})

    @app.put("/api/platform/sop")
    def save_sop():
        try:
            saved = store.update_sop(request.get_json(force=True))
            return jsonify({"ok": True, "data": saved})
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "message": str(exc)}), 400

    @app.post("/api/platform/sop/releases")
    def publish_sop():
        try:
            return jsonify({"ok": True, "data": store.publish_sop()}), 201
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "message": str(exc)}), 400

    @app.post("/api/platform/deployments")
    def create_deployment():
        payload = request.get_json(silent=True) or {}
        try:
            deployment = store.deploy_release(
                release_id=payload.get("releaseId", ""),
                station_id=payload.get("stationId", ""),
            )
            return jsonify({"ok": True, "data": deployment}), 201
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "message": str(exc)}), 400

    @app.get("/api/platform/health")
    def health():
        runtime_health = runtime.health()
        camera_health = _probe(camera_health_url)
        return jsonify(
            {
                "ok": True,
                "data": {
                    "platform": {"online": True, "message": "Platform API ready"},
                    "runtime": {
                        "online": runtime_health["online"],
                        "message": runtime_health["message"],
                    },
                    "camera": {
                        **camera_health,
                        "streamUrl": camera_url,
                    },
                    "inference": {
                        "online": bool(
                            runtime_health.get("runtime")
                            and runtime_health["runtime"].get("task")
                            and runtime_health["runtime"]["task"].get("runtimeFps")
                        ),
                        "message": (
                            "Receiving inference results"
                            if runtime_health.get("runtime")
                            and runtime_health["runtime"].get("task")
                            and runtime_health["runtime"]["task"].get("runtimeFps")
                            else "Waiting for runtime telemetry"
                        ),
                    },
                },
            }
        )

    @app.get("/api/platform/runtime")
    def current_runtime():
        try:
            return jsonify({"ok": True, "data": runtime.current()})
        except RuntimeUnavailable as exc:
            return jsonify({"ok": False, "message": str(exc), "data": None}), 503

    @app.post("/api/platform/runtime/<command>")
    def control_runtime(command: str):
        if command not in {"start", "stop", "reset"}:
            return jsonify({"ok": False, "message": "unknown command"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            data = getattr(runtime, command)(payload.get("taskCode"))
            return jsonify({"ok": True, "data": data})
        except RuntimeUnavailable as exc:
            return jsonify({"ok": False, "message": str(exc)}), 503

    return app
