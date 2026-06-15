import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app


class ApiGuardTests(unittest.TestCase):
    def setUp(self):
        app._request_history.clear()
        self.client = TestClient(app.app)

    def test_health_is_not_rate_limited(self):
        for _ in range(app._rate_limit_per_minute + 2):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)

    def test_cors_rejects_unknown_origin(self):
        response = self.client.options(
            "/api/run-strategy",
            headers={
                "Access-Control-Request-Method": "POST",
                "Origin": "https://attacker.example",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_strategy_text_length_is_limited(self):
        response = self.client.post(
            "/api/run-strategy",
            json={"strateji_metni": "A" * 501},
        )
        self.assertEqual(response.status_code, 422)

    def test_request_body_size_is_limited(self):
        response = self.client.post(
            "/api/run-strategy",
            content=b"x" * (app._max_body_bytes + 1),
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "request_body_too_large")

    def test_rate_limit_rejects_excess_requests(self):
        with patch.object(app, "_rate_limit_per_minute", 1):
            first = self.client.post("/api/run-strategy", json={})
            second = self.client.post("/api/run-strategy", json={})

        self.assertEqual(first.status_code, 422)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers["retry-after"], "60")

    def test_capacity_limit_rejects_when_full(self):
        for _ in range(app._max_concurrent_strategies):
            app._strategy_slots.acquire()
        try:
            response = self.client.post("/api/run-strategy", json={})
        finally:
            for _ in range(app._max_concurrent_strategies):
                app._strategy_slots.release()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "strategy_capacity_exceeded")


if __name__ == "__main__":
    unittest.main()
