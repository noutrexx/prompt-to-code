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


class CompareEndpointTests(unittest.TestCase):
    def setUp(self):
        app._request_history.clear()
        self.client = TestClient(app.app)

    def test_too_many_symbols_rejected(self):
        # Pydantic max_length=5 dogrulamasi (ag cagrisi yapilmadan)
        response = self.client.post(
            "/api/compare-strategy",
            json={"strateji_metni": "RSI 30 altinda al", "semboller": ["A", "B", "C", "D", "E", "F"]},
        )
        self.assertEqual(response.status_code, 422)

    def test_empty_symbols_rejected(self):
        response = self.client.post(
            "/api/compare-strategy",
            json={"strateji_metni": "RSI 30 altinda al", "semboller": []},
        )
        self.assertEqual(response.status_code, 422)

    def test_compare_is_rate_limited(self):
        # Korumali yollar arasinda; limit asilinca 429 doner.
        with patch.object(app, "_rate_limit_per_minute", 1):
            first = self.client.post("/api/compare-strategy", json={})
            second = self.client.post("/api/compare-strategy", json={})
        # Ilk istek govde dogrulamasinda (422) takilir ama limiti tuketir; ikinci 429.
        self.assertEqual(second.status_code, 429)

    def test_compare_happy_path_returns_ranked_results(self):
        fake_rule = type("R", (), {"model_dump": lambda self, mode=None: {"conditions": [], "action": "BUY"}})()
        ranked = [{"symbol": "AAA", "ok": True, "rank": 1, "metrics": {"toplam_kar_zarar_pct": 10.0}}]
        with patch.object(app._parser, "parse", return_value=fake_rule), \
             patch.object(app, "compare_symbols", return_value=ranked) as mocked:
            response = self.client.post(
                "/api/compare-strategy",
                json={"strateji_metni": "RSI 30 altinda al", "semboller": ["aaa"]},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["results"], ranked)
        # Sembol normalize edilerek (uppercase) compare_symbols'a iletilmeli
        called_symbols = mocked.call_args.args[0]
        self.assertEqual(called_symbols, ["AAA"])


if __name__ == "__main__":
    unittest.main()
