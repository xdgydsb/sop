from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "apps" / "api"))

from platform_api import create_app
from platform_api.store import CatalogStore


class FakeRuntime:
    def __init__(self) -> None:
        self.mode = "READY"

    def health(self):
        return {"online": True, "message": "connected", "runtime": self.current()}

    def current(self):
        return {
            "task": {"taskCode": "TASK-1", "runtimeMode": self.mode, "runtimeFps": 20},
            "steps": [],
            "recentEvents": [],
            "recentAlarms": [],
        }

    def start(self, task_code=None):
        self.mode = "ARMED"
        return self.current()

    def stop(self, task_code=None):
        self.mode = "STOPPED"
        return self.current()

    def reset(self, task_code=None):
        self.mode = "READY"
        return self.current()


class PlatformApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        store = CatalogStore(Path(self.temp.name) / "catalog.json")
        self.app = create_app(FakeRuntime(), store)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_serves_product_page_and_catalog(self) -> None:
        with self.client.get("/") as page:
            self.assertEqual(200, page.status_code)
            self.assertIn("工业作业智能平台", page.get_data(as_text=True))
        catalog = self.client.get("/api/platform/catalog").get_json()
        self.assertTrue(catalog["ok"])
        self.assertEqual(5, len(catalog["data"]["sop"]["steps"]))

    def test_runtime_controls_use_adapter(self) -> None:
        response = self.client.post("/api/platform/runtime/start", json={})
        self.assertEqual(200, response.status_code)
        self.assertEqual("ARMED", response.get_json()["data"]["task"]["runtimeMode"])

    def test_sop_draft_is_validated_and_persisted(self) -> None:
        sop = self.client.get("/api/platform/catalog").get_json()["data"]["sop"]
        sop["steps"][0]["holdMs"] = 420
        response = self.client.put("/api/platform/sop", json=sop)
        self.assertEqual(200, response.status_code)
        saved = self.client.get("/api/platform/catalog").get_json()["data"]["sop"]
        self.assertEqual(420, saved["steps"][0]["holdMs"])

    def test_invalid_sop_is_rejected(self) -> None:
        response = self.client.put("/api/platform/sop", json={"steps": []})
        self.assertEqual(400, response.status_code)

    def test_sop_release_is_immutable_and_can_be_deployed(self) -> None:
        release_response = self.client.post("/api/platform/sop/releases")
        self.assertEqual(201, release_response.status_code)
        release = release_response.get_json()["data"]
        self.assertEqual("package-five-step-v1", release["releaseId"])
        self.assertEqual("RELEASED", release["status"])

        draft = self.client.get("/api/platform/catalog").get_json()["data"]["sop"]
        draft["steps"][0]["name"] = "Changed after release"
        self.client.put("/api/platform/sop", json=draft)

        catalog = self.client.get("/api/platform/catalog").get_json()["data"]
        self.assertEqual("打开盒子", catalog["releases"][0]["steps"][0]["name"])

        deploy_response = self.client.post(
            "/api/platform/deployments",
            json={
                "releaseId": release["releaseId"],
                "stationId": "STATION-01",
            },
        )
        self.assertEqual(201, deploy_response.status_code)
        deployment = deploy_response.get_json()["data"]
        self.assertEqual("ACTIVE", deployment["status"])
        self.assertEqual(release["releaseId"], deployment["releaseId"])

    def test_unknown_release_cannot_be_deployed(self) -> None:
        response = self.client.post(
            "/api/platform/deployments",
            json={"releaseId": "missing", "stationId": "STATION-01"},
        )
        self.assertEqual(400, response.status_code)


if __name__ == "__main__":
    unittest.main()
