"""SQLite-backed catalog for SOP drafts, releases, and deployments."""

from __future__ import annotations

import json
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


LEGACY_STAGE3_EVENTS = [
    "box_opened",
    "earphone_in_box",
    "charger_in_box",
    "green_bag_in_box",
    "box_closed",
]

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
}


class CatalogStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("platform.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workstation (
                    station_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sop_draft (
                    sop_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    station_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sop_draft_step (
                    sop_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_no INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    event_code TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    hold_ms INTEGER NOT NULL,
                    PRIMARY KEY (sop_id, step_id),
                    UNIQUE (sop_id, step_no),
                    UNIQUE (sop_id, event_code),
                    FOREIGN KEY (sop_id) REFERENCES sop_draft(sop_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sop_release (
                    release_id TEXT PRIMARY KEY,
                    sop_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    compatibility TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    released_at TEXT NOT NULL,
                    UNIQUE (sop_id, version)
                );
                CREATE TABLE IF NOT EXISTS deployment (
                    deployment_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    runtime_applied INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (release_id) REFERENCES sop_release(release_id)
                );
                """
            )
            count = db.execute("SELECT COUNT(*) FROM sop_draft").fetchone()[0]
            if count == 0:
                self._seed(db)

    def _seed(self, db: sqlite3.Connection) -> None:
        for model in DEFAULT_CATALOG["models"]:
            db.execute(
                "INSERT INTO model_registry(model_id, payload_json) VALUES (?, ?)",
                (model["id"], json.dumps(model, ensure_ascii=False)),
            )
        for station in DEFAULT_CATALOG["stations"]:
            db.execute(
                "INSERT INTO workstation(station_id, payload_json) VALUES (?, ?)",
                (station["id"], json.dumps(station, ensure_ascii=False)),
            )
        self._write_draft(db, DEFAULT_CATALOG["sop"])

    def get(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            models = [
                json.loads(row["payload_json"])
                for row in db.execute(
                    "SELECT payload_json FROM model_registry ORDER BY model_id"
                )
            ]
            stations = [
                json.loads(row["payload_json"])
                for row in db.execute(
                    "SELECT payload_json FROM workstation ORDER BY station_id"
                )
            ]
            sop = self._read_draft(db)
            releases = [
                json.loads(row["snapshot_json"])
                for row in db.execute(
                    "SELECT snapshot_json FROM sop_release ORDER BY version"
                )
            ]
            deployments = [
                json.loads(row["payload_json"])
                for row in db.execute(
                    "SELECT payload_json FROM deployment ORDER BY created_at"
                )
            ]
            return {
                "models": models,
                "stations": stations,
                "sop": sop,
                "releases": releases,
                "deployments": deployments,
            }

    def update_sop(self, sop: dict[str, Any]) -> dict[str, Any]:
        self._validate_sop(sop)
        with self._lock, self._connect() as db:
            self._write_draft(db, sop)
        return deepcopy(sop)

    def publish_sop(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            sop = self._read_draft(db)
            self._validate_sop(sop)
            version = (
                db.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM sop_release WHERE sop_id = ?",
                    (sop["id"],),
                ).fetchone()[0]
            )
            compatibility = self._runtime_compatibility(sop)
            released_at = datetime.now(timezone.utc).isoformat()
            release = {
                **sop,
                "releaseId": f"{sop['id']}-v{version}",
                "version": version,
                "status": "RELEASED",
                "runtimeCompatibility": compatibility,
                "releasedAt": released_at,
            }
            db.execute(
                """
                INSERT INTO sop_release(
                    release_id, sop_id, version, compatibility,
                    snapshot_json, released_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    release["releaseId"],
                    sop["id"],
                    version,
                    compatibility,
                    json.dumps(release, ensure_ascii=False),
                    released_at,
                ),
            )
            sop["version"] = version + 1
            sop["status"] = "DRAFT"
            self._write_draft(db, sop)
            return release

    def deploy_release(self, release_id: str, station_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            release_row = db.execute(
                "SELECT snapshot_json, compatibility FROM sop_release WHERE release_id = ?",
                (release_id,),
            ).fetchone()
            if release_row is None:
                raise ValueError("released SOP version not found")
            station = db.execute(
                "SELECT 1 FROM workstation WHERE station_id = ?",
                (station_id,),
            ).fetchone()
            if station is None:
                raise ValueError("station not found")

            release = json.loads(release_row["snapshot_json"])
            compatible = release_row["compatibility"] == "LEGACY_STAGE3"
            status = "STAGED" if compatible else "BLOCKED"
            message = (
                "部署配置已生成，等待 Worker 下发接口确认"
                if compatible
                else "当前 stage3 仅支持固定五步；该版本需要通用 Runtime"
            )
            deployment = {
                "deploymentId": uuid4().hex,
                "releaseId": release_id,
                "stationId": station_id,
                "modelId": release["modelId"],
                "status": status,
                "runtimeApplied": False,
                "message": message,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
            db.execute(
                """
                INSERT INTO deployment(
                    deployment_id, release_id, station_id, status,
                    runtime_applied, payload_json, created_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    deployment["deploymentId"],
                    release_id,
                    station_id,
                    status,
                    json.dumps(deployment, ensure_ascii=False),
                    deployment["createdAt"],
                ),
            )
            return deployment

    def _read_draft(self, db: sqlite3.Connection) -> dict[str, Any]:
        row = db.execute("SELECT * FROM sop_draft ORDER BY sop_id LIMIT 1").fetchone()
        if row is None:
            raise ValueError("SOP draft not found")
        steps = [
            {
                "id": step["step_id"],
                "name": step["name"],
                "event": step["event_code"],
                "rule": step["rule_text"],
                "holdMs": step["hold_ms"],
            }
            for step in db.execute(
                "SELECT * FROM sop_draft_step WHERE sop_id = ? ORDER BY step_no",
                (row["sop_id"],),
            )
        ]
        return {
            "id": row["sop_id"],
            "name": row["name"],
            "version": row["version"],
            "status": row["status"],
            "modelId": row["model_id"],
            "stationId": row["station_id"],
            "steps": steps,
        }

    def _write_draft(self, db: sqlite3.Connection, sop: dict[str, Any]) -> None:
        db.execute(
            """
            INSERT INTO sop_draft(
                sop_id, name, version, status, model_id, station_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sop_id) DO UPDATE SET
                name = excluded.name,
                version = excluded.version,
                status = excluded.status,
                model_id = excluded.model_id,
                station_id = excluded.station_id
            """,
            (
                sop["id"],
                sop["name"],
                int(sop.get("version", 1)),
                sop.get("status", "DRAFT"),
                sop["modelId"],
                sop["stationId"],
            ),
        )
        db.execute("DELETE FROM sop_draft_step WHERE sop_id = ?", (sop["id"],))
        for step_no, step in enumerate(sop["steps"], start=1):
            db.execute(
                """
                INSERT INTO sop_draft_step(
                    sop_id, step_id, step_no, name, event_code, rule_text, hold_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sop["id"],
                    step["id"],
                    step_no,
                    step["name"],
                    step["event"],
                    step["rule"],
                    int(step["holdMs"]),
                ),
            )

    def _validate_sop(self, sop: dict[str, Any]) -> None:
        if not isinstance(sop.get("steps"), list) or not sop["steps"]:
            raise ValueError("SOP must contain at least one step")
        if not sop.get("id") or not sop.get("name"):
            raise ValueError("SOP id and name are required")
        with self._connect() as db:
            if (
                db.execute(
                    "SELECT 1 FROM model_registry WHERE model_id = ?",
                    (sop.get("modelId"),),
                ).fetchone()
                is None
            ):
                raise ValueError("bound model does not exist")
            if (
                db.execute(
                    "SELECT 1 FROM workstation WHERE station_id = ?",
                    (sop.get("stationId"),),
                ).fetchone()
                is None
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

    @staticmethod
    def _runtime_compatibility(sop: dict[str, Any]) -> str:
        events = [step["event"] for step in sop["steps"]]
        return (
            "LEGACY_STAGE3"
            if events == LEGACY_STAGE3_EVENTS
            else "GENERIC_RUNTIME_REQUIRED"
        )
