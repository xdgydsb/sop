"""Small JSON store for the first product slice's editable SOP draft."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_CATALOG = {
    "models": [
        {
            "id": "stage3-package-v1",
            "name": "包装工序视觉模型",
            "version": "1.0.0",
            "type": "目标检测 + 手物关系 + 时序规则",
            "status": "BASELINE_VERIFIED",
            "verifiedAt": "2026-06-11",
            "verification": "真实海康相机五步流程稳定基线",
            "labels": ["box_open", "box_closed", "earphone", "charger", "green_bag"],
        }
    ],
    "stations": [
        {
            "id": "STATION-01",
            "name": "包装检测工位 01",
            "camera": "MV-CS050-10UC",
            "worker": "Local Hik Camera Worker",
            "inference": "192.168.31.19:8765",
        }
    ],
    "sop": {
        "id": "package-five-step",
        "name": "包装五步作业",
        "version": 1,
        "status": "DRAFT",
        "modelId": "stage3-package-v1",
        "stationId": "STATION-01",
        "steps": [
            {
                "id": "open_box",
                "name": "打开盒子",
                "event": "box_opened",
                "rule": "关盒 → 手部交互 → 开盒并稳定",
                "holdMs": 300,
            },
            {
                "id": "place_earphone",
                "name": "放入耳机",
                "event": "earphone_in_box",
                "rule": "盒外 → 手物交互 → 盒内并留存",
                "holdMs": 300,
            },
            {
                "id": "place_charger",
                "name": "放入充电器",
                "event": "charger_in_box",
                "rule": "盒外 → 手物交互 → 盒内并留存",
                "holdMs": 300,
            },
            {
                "id": "place_green_bag",
                "name": "放入绿色小袋",
                "event": "green_bag_in_box",
                "rule": "盒外 → 手物交互 → 盒内并留存",
                "holdMs": 300,
            },
            {
                "id": "close_box",
                "name": "关闭盒子",
                "event": "box_closed",
                "rule": "物料齐全 → 手部交互 → 关盒并稳定",
                "holdMs": 300,
            },
        ],
    },
    "releases": [],
    "deployments": [],
}


class CatalogStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = deepcopy(DEFAULT_CATALOG)
        if self.path and self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            self._data.setdefault("releases", [])
            self._data.setdefault("deployments", [])
            for model in self._data.get("models", []):
                if model.get("status") == "READY":
                    model["status"] = "BASELINE_VERIFIED"
                    model["verifiedAt"] = "2026-06-11"
                    model["verification"] = "真实海康相机五步流程稳定基线"

    def get(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def update_sop(self, sop: dict[str, Any]) -> dict[str, Any]:
        self._validate_sop(sop)

        with self._lock:
            self._data["sop"] = deepcopy(sop)
            self._persist()
            return deepcopy(self._data["sop"])

    def publish_sop(self) -> dict[str, Any]:
        with self._lock:
            sop = deepcopy(self._data["sop"])
            self._validate_sop(sop)
            version = max(
                (release["version"] for release in self._data["releases"]),
                default=0,
            ) + 1
            release = {
                **sop,
                "releaseId": f"{sop['id']}-v{version}",
                "version": version,
                "status": "RELEASED",
                "releasedAt": datetime.now(timezone.utc).isoformat(),
            }
            self._data["releases"].append(release)
            self._data["sop"]["version"] = version + 1
            self._data["sop"]["status"] = "DRAFT"
            self._persist()
            return deepcopy(release)

    def deploy_release(self, release_id: str, station_id: str) -> dict[str, Any]:
        with self._lock:
            release = next(
                (
                    item
                    for item in self._data["releases"]
                    if item["releaseId"] == release_id
                ),
                None,
            )
            if release is None:
                raise ValueError("released SOP version not found")
            if not any(item["id"] == station_id for item in self._data["stations"]):
                raise ValueError("station not found")

            for deployment in self._data["deployments"]:
                if deployment["stationId"] == station_id:
                    deployment["status"] = "REPLACED"

            deployment = {
                "deploymentId": uuid4().hex,
                "releaseId": release_id,
                "stationId": station_id,
                "modelId": release["modelId"],
                "status": "STAGED",
                "runtimeApplied": False,
                "message": "部署配置已生成，等待 Worker 下发接口确认",
                "stagedAt": datetime.now(timezone.utc).isoformat(),
            }
            self._data["deployments"].append(deployment)
            self._persist()
            return deepcopy(deployment)

    def _validate_sop(self, sop: dict[str, Any]) -> None:
        if not isinstance(sop.get("steps"), list) or not sop["steps"]:
            raise ValueError("SOP must contain at least one step")
        if not sop.get("id") or not sop.get("name"):
            raise ValueError("SOP id and name are required")
        if not any(model["id"] == sop.get("modelId") for model in self._data["models"]):
            raise ValueError("bound model does not exist")
        if not any(
            station["id"] == sop.get("stationId")
            for station in self._data["stations"]
        ):
            raise ValueError("bound station does not exist")
        required = {"id", "name", "event", "rule", "holdMs"}
        step_ids: set[str] = set()
        events: set[str] = set()
        for index, step in enumerate(sop["steps"], start=1):
            missing = required - set(step)
            if missing:
                raise ValueError(
                    f"step {index} missing fields: {', '.join(sorted(missing))}"
                )
            if int(step["holdMs"]) < 0:
                raise ValueError(f"step {index} holdMs must be non-negative")
            if not str(step["name"]).strip() or not str(step["rule"]).strip():
                raise ValueError(f"step {index} name and rule are required")
            if step["id"] in step_ids:
                raise ValueError(f"duplicate step id: {step['id']}")
            if step["event"] in events:
                raise ValueError(f"duplicate step event: {step['event']}")
            step_ids.add(step["id"])
            events.add(step["event"])

    def _persist(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
