"""
test_backtest_engine.py
-----------------------
Backtester icin agdan bagimsiz, deterministik birim testleri.
Sentetik OHLC + indikator verisi kullanir; yfinance/Gemini cagrilmaz.
"""

import unittest

import numpy as np
import pandas as pd

from backtest_engine import Backtester, BacktestError


def _make_df() -> pd.DataFrame:
    """6 barlik kontrollu veri. RSI sinyali fiyattan bagimsiz tutulur."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            # Acilis ve kapanis kasten farkli -> hangi fiyattan islem yapildigi netlesir
            "Open":  [100, 101, 102, 103, 104, 105],
            "Close": [200, 201, 202, 203, 204, 205],
            # RSI: bar 1'de asiri satim (AL sinyali), bar 4'te asiri alim (SAT sinyali)
            "RSI_14": [50, 25, 50, 50, 75, 50],
        },
        index=dates,
    )


BUY_RULE = {"conditions": [{"indicator": "RSI", "operator": "less_than", "value": 30}]}
SELL_RULE = {"conditions": [{"indicator": "RSI", "operator": "greater_than", "value": 70}]}


class NextOpenExecutionTests(unittest.TestCase):
    """Varsayilan next_open: sinyal barinin BIR SONRAKI acilisindan islem."""

    def setUp(self):
        self.bt = Backtester(_make_df(), BUY_RULE, SELL_RULE)  # execution="next_open" (varsayilan)
        self.bt.run()

    def test_entry_uses_next_bar_open(self):
        # AL sinyali bar 1 (RSI=25) -> islem bar 2 acilisi = 102
        self.assertEqual(self.bt.trades[0]["entry_price"], 102)

    def test_exit_uses_next_bar_open(self):
        # SAT sinyali bar 4 (RSI=75) -> islem bar 5 acilisi = 105
        self.assertEqual(self.bt.trades[0]["exit_price"], 105)

    def test_single_completed_trade(self):
        self.assertEqual(len(self.bt.trades), 1)
        self.assertEqual(self.bt.metrics["toplam_islem_sayisi"], 1)

    def test_pnl_pct_matches_open_to_open(self):
        expected = round((105 - 102) / 102 * 100.0, 4)
        self.assertEqual(self.bt.trades[0]["pnl_pct"], expected)

    def test_no_lookahead_signal_not_same_bar_close(self):
        # Eski hatali davranista giris 202 (sinyal barinin kapanisi) olurdu.
        self.assertNotEqual(self.bt.trades[0]["entry_price"], 202)


class CloseExecutionTests(unittest.TestCase):
    """execution='close': eski (look-ahead'li) davranis korunabilmeli."""

    def test_entry_and_exit_use_same_bar_close(self):
        bt = Backtester(_make_df(), BUY_RULE, SELL_RULE, execution="close")
        bt.run()
        # AL sinyali bar 1 kapanisi = 201, SAT sinyali bar 4 kapanisi = 204
        self.assertEqual(bt.trades[0]["entry_price"], 201)
        self.assertEqual(bt.trades[0]["exit_price"], 204)


class CommissionAndBalanceTests(unittest.TestCase):
    def test_final_balance_accounts_for_commission(self):
        bt = Backtester(_make_df(), BUY_RULE, SELL_RULE, initial_balance=10_000, commission=0.001)
        bt.run()
        shares = (10_000 * 0.999) / 102          # giris (acilis 102)
        expected_final = shares * 105 * 0.999     # cikis (acilis 105)
        self.assertAlmostEqual(bt.metrics["son_bakiye"], round(expected_final, 2), places=2)


class EdgeCaseTests(unittest.TestCase):
    def test_no_signals_means_no_trades(self):
        df = _make_df()
        df["RSI_14"] = 50  # hicbir kosul tetiklenmez
        bt = Backtester(df, BUY_RULE, SELL_RULE)
        bt.run()
        self.assertEqual(len(bt.trades), 0)
        self.assertEqual(bt.metrics["son_bakiye"], 10_000.0)

    def test_open_position_closed_at_last_bar(self):
        df = _make_df()
        df["RSI_14"] = [50, 25, 50, 50, 50, 50]  # AL var, SAT yok
        bt = Backtester(df, BUY_RULE, SELL_RULE)
        bt.run()
        self.assertEqual(len(bt.trades), 1)
        # Acik pozisyon son barin KAPANISINDAN kapatilir (mark-to-market tasfiye)
        self.assertEqual(bt.trades[0]["exit_price"], 205)
        self.assertIn("note", bt.trades[0])

    def test_invalid_execution_raises(self):
        with self.assertRaises(BacktestError):
            Backtester(_make_df(), BUY_RULE, SELL_RULE, execution="future")

    def test_empty_df_raises(self):
        with self.assertRaises(BacktestError):
            Backtester(pd.DataFrame(), BUY_RULE)


class MetricsTests(unittest.TestCase):
    def test_buy_hold_uses_first_and_last_close(self):
        bt = Backtester(_make_df(), BUY_RULE, SELL_RULE)
        bt.run()
        # Close: 200 -> 205  =>  %2.5
        self.assertEqual(bt.metrics["al_tut_getiri_pct"], 2.5)

    def test_alpha_is_strategy_minus_buy_hold(self):
        bt = Backtester(_make_df(), BUY_RULE, SELL_RULE)
        bt.run()
        expected = round(bt.metrics["toplam_kar_zarar_pct"] - bt.metrics["al_tut_getiri_pct"], 2)
        self.assertEqual(bt.metrics["strateji_alpha_pct"], expected)

    def test_sharpe_key_present_and_numeric(self):
        bt = Backtester(_make_df(), BUY_RULE, SELL_RULE)
        bt.run()
        self.assertIn("sharpe_orani", bt.metrics)
        self.assertIsInstance(bt.metrics["sharpe_orani"], float)

    def test_flat_equity_has_zero_sharpe(self):
        df = _make_df()
        df["RSI_14"] = 50  # hic islem yok -> equity sabit
        bt = Backtester(df, BUY_RULE, SELL_RULE)
        bt.run()
        self.assertEqual(bt.metrics["sharpe_orani"], 0.0)


class ConditionOperatorTests(unittest.TestCase):
    """crosses_above/below operatorleri bir onceki bara gore kesisim arar."""

    def _df_with_macd(self, macd_values):
        dates = pd.date_range("2024-01-01", periods=len(macd_values), freq="D")
        return pd.DataFrame(
            {
                "Open": np.arange(100, 100 + len(macd_values), dtype=float),
                "Close": np.arange(100, 100 + len(macd_values), dtype=float),
                "MACD": macd_values,
            },
            index=dates,
        )

    def test_crosses_above_zero_fires_once(self):
        # MACD -1 -> +1 gecisi yalnizca bar 2'de yukari kesisir
        df = self._df_with_macd([-2.0, -1.0, 1.0, 2.0])
        rule = {"conditions": [{"indicator": "MACD", "operator": "crosses_above", "value": 0}]}
        bt = Backtester(df, rule)
        signals = bt._evaluate_rule(bt.buy_rule)
        self.assertEqual(signals.sum(), 1)
        self.assertTrue(bool(signals.iloc[2]))

    def test_crosses_below_zero_fires_once(self):
        df = self._df_with_macd([2.0, 1.0, -1.0, -2.0])
        rule = {"conditions": [{"indicator": "MACD", "operator": "crosses_below", "value": 0}]}
        bt = Backtester(df, rule)
        signals = bt._evaluate_rule(bt.buy_rule)
        self.assertEqual(signals.sum(), 1)
        self.assertTrue(bool(signals.iloc[2]))


if __name__ == "__main__":
    unittest.main()
