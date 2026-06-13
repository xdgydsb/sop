"""Adapter for the useful runtime endpoints in the existing RuoYi backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class RuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyRuntimeConfig:
    base_url: str = "http://127.0.0.1:8080"
    product_id: int = 100
    sop_id: int = 100
    station_code: str = "STATION-01"
    camera_code: str = "MV-CS050-10UC"
    timeout_seconds: float = 1.5


class LegacyRuntimeAdapter:
    def __init__(self, config: LegacyRuntimeConfig | None = None) -> None:
        self.config = config or LegacyRuntimeConfig()

    def health(self) -> dict[str, Any]:
        try:
            runtime = self.current()
            return {
                "online": True,
                "message": "RuoYi runtime connected",
                "runtime": runtime,
            }
        except RuntimeUnavailable as exc:
            return {"online": False, "message": str(exc), "runtime": None}

    def current(self) -> dict[str, Any] | None:
        query = urllib.parse.urlencode(
            {
                "productId": self.config.product_id,
                "sopId": self.config.sop_id,
            }
        )
        response = self._request(f"/sop/runtime/current?{query}")
        return response.get("data")

    def start(self, task_code: str | None = None) -> dict[str, Any]:
        return self._control("/sop/runtime/session/start", task_code)

    def stop(self, task_code: str | None = None) -> dict[str, Any]:
        return self._control("/sop/runtime/session/stop", task_code)

    def reset(self, task_code: str | None = None) -> dict[str, Any]:
        return self._control("/sop/runtime/session/reset", task_code)

    def _control(self, path: str, task_code: str | None) -> dict[str, Any]:
        payload = {
            "productId": self.config.product_id,
            "sopId": self.config.sop_id,
            "stationCode": self.config.station_code,
            "cameraCode": self.config.camera_code,
            "operatorName": "platform",
        }
        if task_code:
            payload["taskCode"] = task_code
        response = self._request(path, payload, method="POST")
        if response.get("code") != 200:
            raise RuntimeUnavailable(response.get("msg") or "runtime command failed")
        return response.get("data") or {}

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeUnavailable(
                f"RuoYi runtime unavailable: {exc}"
            ) from exc
