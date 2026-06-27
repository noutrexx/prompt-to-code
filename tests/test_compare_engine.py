"""
test_compare_engine.py
----------------------
compare_symbols icin agdan bagimsiz testler. Veri kaynagi (fetcher)
enjekte edilerek her sembole farkli sentetik fiyat serisi verilir.
"""

import unittest

import pandas as pd

from compare_engine import compare_symbols

BUY_RULE = {"conditions": [{"indicator": "RSI", "operator": "less_than", "value": 30}]}
SELL_RULE = {"conditions": [{"indicator": "RSI", "operator": "greater_than", "value": 70}]}


def _df(open_last: float):
    """6 barlik veri; cikis acilisi open_last ile getiriyi kontrol ederiz."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "Open":  [100, 100, 100, 100, 100, open_last],
            "High":  [100, 100, 100, 100, 100, open_last],
            "Low":   [100, 100, 100, 100, 100, open_last],
            "Close": [100, 100, 100, 100, 100, open_last],
            # AL bar 1 -> giris bar 2 acilisi (100); SAT bar 4 -> cikis bar 5 acilisi (open_last)
            "RSI_14": [50, 25, 50, 50, 75, 50],
        },
        index=dates,
    )


class CompareSymbolsTests(unittest.TestCase):
    def setUp(self):
        # AAA daha karli (cikis 130) > BBB (cikis 110)
        self._data = {"AAA": _df(130), "BBB": _df(110)}
        self.fetcher = lambda sym: self._data[sym]

    def test_ranks_by_total_return_desc(self):
        rows = compare_symbols(["BBB", "AAA"], BUY_RULE, SELL_RULE, fetcher=self.fetcher)
        self.assertEqual(rows[0]["symbol"], "AAA")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["symbol"], "BBB")
        self.assertEqual(rows[1]["rank"], 2)

    def test_each_row_has_metrics(self):
        rows = compare_symbols(["AAA", "BBB"], BUY_RULE, SELL_RULE, fetcher=self.fetcher)
        for row in rows:
            self.assertTrue(row["ok"])
            self.assertIn("toplam_kar_zarar_pct", row["metrics"])

    def test_failed_symbol_is_isolated_and_last(self):
        def fetcher(sym):
            if sym == "BAD":
                raise RuntimeError("veri yok")
            return self._data[sym]

        rows = compare_symbols(["AAA", "BAD"], BUY_RULE, SELL_RULE, fetcher=fetcher)
        good = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        self.assertEqual(len(good), 1)
        self.assertEqual(bad[0]["symbol"], "BAD")
        self.assertIn("error", bad[0])
        self.assertEqual(rows[-1]["symbol"], "BAD")  # hatali en sonda

    def test_duplicate_symbols_collapsed(self):
        rows = compare_symbols(["AAA", "AAA"], BUY_RULE, SELL_RULE, fetcher=self.fetcher)
        self.assertEqual(len(rows), 1)

    def test_empty_symbols_raises(self):
        with self.assertRaises(ValueError):
            compare_symbols([], BUY_RULE, fetcher=self.fetcher)

    def test_backtest_kwargs_forwarded(self):
        # stop_loss gibi kwarg'lar Backtester'a iletilir (hata vermemeli)
        rows = compare_symbols(["AAA"], BUY_RULE, SELL_RULE, fetcher=self.fetcher, stop_loss=0.05)
        self.assertTrue(rows[0]["ok"])


if __name__ == "__main__":
    unittest.main()
