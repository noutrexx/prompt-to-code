"""
backtest_engine.py
------------------
data_engine.py (fiyat + indikator verisi) ve nlp_parser.py (kural semasi)
ciktilarini kullanarak gecmise donuk test (backtest) yapan motor.

Kullanim:
    from data_engine import MarketDataFetcher
    from backtest_engine import Backtester

    df = MarketDataFetcher("THYAO.IS", period="2y").run()
    buy_rule  = {"asset": "THYAO.IS",
                 "conditions": [{"indicator": "RSI", "operator": "less_than", "value": 35}],
                 "action": "BUY"}
    sell_rule = {"asset": "THYAO.IS",
                 "conditions": [{"indicator": "RSI", "operator": "greater_than", "value": 65}],
                 "action": "SELL"}
    bt = Backtester(df, buy_rule, sell_rule)
    sonuc = bt.run()
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BacktestError(Exception):
    """Backtest sirasinda olusan anlamli hata."""


class Backtester:
    """Bir kural seti ile DataFrame uzerinde gecmise donuk simulasyon yapar."""

    # nlp_parser indikator adi -> DataFrame kolon adi eslemesi (buyuk harf anahtarlar)
    INDICATOR_MAP = {
        # Fiyat
        "PRICE": "Close", "CLOSE": "Close", "OPEN": "Open", "HIGH": "High", "LOW": "Low",
        # Momentum
        "RSI": "RSI_14", "RSI_14": "RSI_14",
        "STOCH": "STOCH_K", "STOCH_K": "STOCH_K", "STOCH_D": "STOCH_D",
        # Trend: SMA / EMA
        "SMA": "SMA_50",
        "SMA_20": "SMA_20", "SMA_50": "SMA_50", "SMA_100": "SMA_100", "SMA_200": "SMA_200",
        "EMA": "EMA_50",
        "EMA_12": "EMA_12", "EMA_26": "EMA_26", "EMA_50": "EMA_50", "EMA_200": "EMA_200",
        # MACD
        "MACD": "MACD", "MACD_SIGNAL": "MACD_Signal", "MACD_HIST": "MACD_Hist",
        # ADX
        "ADX": "ADX_14", "ADX_14": "ADX_14",
        # Bollinger
        "BB_UPPER": "BB_Upper", "BB_MIDDLE": "BB_Middle", "BB_LOWER": "BB_Lower",
        # Hacim
        "VOLUME": "Volume", "VOLUME_SMA": "Volume_SMA_20", "VOLUME_SMA_20": "Volume_SMA_20",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        buy_rule: Union[Dict[str, Any], str],
        sell_rule: Optional[Union[Dict[str, Any], str]] = None,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,  # %0.1
    ) -> None:
        if df is None or df.empty:
            raise BacktestError("Bos veya gecersiz DataFrame.")
        if "Close" not in df.columns:
            raise BacktestError("DataFrame 'Close' kolonu icermeli.")

        self.df = df.copy()
        self.buy_rule = self._as_dict(buy_rule)
        self.sell_rule = self._as_dict(sell_rule) if sell_rule is not None else None
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)

        # Sonuc taşiyicilari
        self.signals: List[Dict[str, Any]] = []   # [{date, side, price}]
        self.trades: List[Dict[str, Any]] = []     # tamamlanmis alis-satis ciftleri
        self.equity_curve: Optional[pd.Series] = None
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Yardimcilar
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_dict(rule: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        if isinstance(rule, str):
            try:
                return json.loads(rule)
            except json.JSONDecodeError as exc:
                raise BacktestError(f"Kural JSON cozumlenemedi: {exc}") from exc
        if isinstance(rule, dict):
            return rule
        raise BacktestError("Kural dict veya JSON string olmalidir.")

    def _column(self, indicator: str) -> str:
        """Indikator adini gercek DataFrame kolonuna cevirir."""
        key = str(indicator).strip().upper()
        col = self.INDICATOR_MAP.get(key, indicator)
        if col not in self.df.columns:
            raise BacktestError(
                f"'{indicator}' icin '{col}' kolonu DataFrame'de yok. "
                f"Mevcut kolonlar: {list(self.df.columns)}"
            )
        return col

    def _operand(self, value: Any) -> Tuple[Any, Any]:
        """
        Kosul 'value' alanini (simdiki, onceki) operand'a cevirir.
        Sayi ise sabit; string ise ilgili kolonun Series'i ve shift(1)'i doner.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), float(value)
        # String -> kolon referansi (orn. "price", "SMA_50")
        series = self.df[self._column(value)]
        return series, series.shift(1)

    # ------------------------------------------------------------------ #
    # Kosul degerlendirme (vektorel)
    # ------------------------------------------------------------------ #
    def _evaluate_condition(self, cond: Dict[str, Any]) -> pd.Series:
        """Tek bir kosulu boolean Series olarak dondurur."""
        left = self.df[self._column(cond["indicator"])]
        left_prev = left.shift(1)
        op = str(cond["operator"]).lower()
        right_now, right_prev = self._operand(cond["value"])

        if op == "less_than":
            result = left < right_now
        elif op == "greater_than":
            result = left > right_now
        elif op == "equals":
            result = np.isclose(left, right_now)
            result = pd.Series(result, index=left.index)
        elif op == "crosses_above":
            result = (left_prev <= right_prev) & (left > right_now)
        elif op == "crosses_below":
            result = (left_prev >= right_prev) & (left < right_now)
        else:
            raise BacktestError(f"Bilinmeyen operator: {cond['operator']}")

        return result.fillna(False)

    def _evaluate_rule(self, rule: Optional[Dict[str, Any]]) -> pd.Series:
        """Kuraldaki TUM kosullar saglandiginda True olan boolean Series."""
        if rule is None:
            return pd.Series(False, index=self.df.index)
        conditions = rule.get("conditions", [])
        if not conditions:
            return pd.Series(False, index=self.df.index)
        mask = pd.Series(True, index=self.df.index)
        for cond in conditions:
            mask &= self._evaluate_condition(cond)
        return mask

    # ------------------------------------------------------------------ #
    # Simulasyon
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        """Backtest'i calistirir; metrikleri hesaplar ve dondurur."""
        buy_signals = self._evaluate_rule(self.buy_rule)

        # Satis kurali verilmediyse alis kuralinin tersini kullan (basit cikis):
        if self.sell_rule is not None:
            sell_signals = self._evaluate_rule(self.sell_rule)
        else:
            sell_signals = pd.Series(False, index=self.df.index)

        close = self.df["Close"].to_numpy()
        dates = self.df.index
        buy_arr = buy_signals.to_numpy()
        sell_arr = sell_signals.to_numpy()

        cash = self.initial_balance
        shares = 0.0
        in_position = False
        entry_price = 0.0
        equity = np.empty(len(close), dtype=float)

        self.signals.clear()
        self.trades.clear()

        for i in range(len(close)):
            price = close[i]

            # Once cikis (satis), sonra giris (alis) — ayni bar iki islem olmaz
            if in_position and sell_arr[i]:
                cash = shares * price * (1.0 - self.commission)
                pnl_pct = (price - entry_price) / entry_price * 100.0
                self.trades.append({
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(float(price), 4),
                    "pnl_pct": round(pnl_pct, 4),
                })
                self.signals.append({
                    "date": self._fmt_date(dates[i]),
                    "side": "SELL",
                    "price": round(float(price), 4),
                })
                shares = 0.0
                in_position = False

            elif (not in_position) and buy_arr[i]:
                shares = (cash * (1.0 - self.commission)) / price
                entry_price = price
                cash = 0.0
                in_position = True
                self.signals.append({
                    "date": self._fmt_date(dates[i]),
                    "side": "BUY",
                    "price": round(float(price), 4),
                })

            # Bar sonu portfoy degeri (mark-to-market)
            equity[i] = cash + shares * price

        # Acik pozisyon kaldiysa son fiyattan kapat (gerceklesmis K/Z icin)
        if in_position:
            last_price = close[-1]
            cash = shares * last_price * (1.0 - self.commission)
            pnl_pct = (last_price - entry_price) / entry_price * 100.0
            self.trades.append({
                "entry_price": round(entry_price, 4),
                "exit_price": round(float(last_price), 4),
                "pnl_pct": round(pnl_pct, 4),
                "note": "acik pozisyon son barda kapatildi",
            })
            self.signals.append({
                "date": self._fmt_date(dates[-1]),
                "side": "SELL",
                "price": round(float(last_price), 4),
            })
            shares = 0.0
            equity[-1] = cash

        self.equity_curve = pd.Series(equity, index=dates, name="Equity")
        self.metrics = self._calculate_metrics(cash)
        return self.metrics

    # ------------------------------------------------------------------ #
    # Metrikler
    # ------------------------------------------------------------------ #
    def _calculate_metrics(self, final_cash: float) -> Dict[str, Any]:
        final_equity = float(self.equity_curve.iloc[-1]) if self.equity_curve is not None else final_cash

        total_return_pct = (final_equity - self.initial_balance) / self.initial_balance * 100.0

        n_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["pnl_pct"] > 0)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        # Maksimum dusus (peak-to-trough)
        eq = self.equity_curve
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100.0) if len(eq) else 0.0

        return {
            "baslangic_bakiye": round(self.initial_balance, 2),
            "son_bakiye": round(final_equity, 2),
            "toplam_kar_zarar_pct": round(total_return_pct, 2),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "toplam_islem_sayisi": n_trades,
            "toplam_sinyal_sayisi": len(self.signals),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt_date(d: Any) -> str:
        try:
            return pd.Timestamp(d).strftime("%Y-%m-%d")
        except Exception:
            return str(d)

    def to_json(self, indent: int = 2) -> str:
        """Metrikleri ve sinyalleri JSON olarak dondurur (frontend icin)."""
        payload = {
            "metrics": self.metrics,
            "signals": self.signals,
            "trades": self.trades,
        }
        return json.dumps(payload, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------- #
# Test blogu
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from data_engine import MarketDataFetcher

    # 2 yillik veri (SMA_200 sonrasi yeterli satir kalmasi icin)
    df = MarketDataFetcher("THYAO.IS", period="2y").run()
    print(f"Veri: {len(df)} satir, {df.index.min().date()} -> {df.index.max().date()}")

    # Manuel kural seti (nlp_parser semasiyla uyumlu)
    buy_rule = {
        "asset": "THYAO.IS",
        "conditions": [{"indicator": "RSI", "operator": "less_than", "value": 35}],
        "action": "BUY",
    }
    sell_rule = {
        "asset": "THYAO.IS",
        "conditions": [{"indicator": "RSI", "operator": "greater_than", "value": 65}],
        "action": "SELL",
    }

    bt = Backtester(df, buy_rule, sell_rule, initial_balance=10_000, commission=0.001)
    metrics = bt.run()

    print("\n=== BACKTEST SONUCLARI (THYAO.IS) ===")
    print("Kural: RSI<35 AL  /  RSI>65 SAT  |  Komisyon: %0.1")
    for k, v in metrics.items():
        print(f"  {k:>22}: {v}")

    print(f"\n=== Uretilen Sinyaller ({len(bt.signals)}) ===")
    for s in bt.signals[:10]:
        print(f"  {s['date']}  {s['side']:>4}  @ {s['price']}")
    if len(bt.signals) > 10:
        print(f"  ... (+{len(bt.signals) - 10} sinyal daha)")
