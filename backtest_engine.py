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
        execution: str = "next_open",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        if df is None or df.empty:
            raise BacktestError("Bos veya gecersiz DataFrame.")
        if "Close" not in df.columns:
            raise BacktestError("DataFrame 'Close' kolonu icermeli.")
        if execution not in ("next_open", "close"):
            raise BacktestError("execution 'next_open' veya 'close' olmalidir.")

        self.df = df.copy()
        self.buy_rule = self._as_dict(buy_rule)
        self.sell_rule = self._as_dict(sell_rule) if sell_rule is not None else None
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        # "next_open": bir barin kapanisinda gozlenen sinyal, look-ahead'i
        # onlemek icin BIR SONRAKI barin acilisinda islem gorur (gercekci).
        # "close": eski davranis — sinyal bariyla ayni kapanistan islem (look-ahead).
        self.execution = execution
        # Koruyucu emirler: giris fiyatina gore oran (0.05 = %5). None ise kapali.
        self.stop_loss = self._validate_fraction(stop_loss, "stop_loss")
        self.take_profit = self._validate_fraction(take_profit, "take_profit")

        # Sonuc taşiyicilari
        self.signals: List[Dict[str, Any]] = []   # [{date, side, price}]
        self.trades: List[Dict[str, Any]] = []     # tamamlanmis alis-satis ciftleri
        self.equity_curve: Optional[pd.Series] = None
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Yardimcilar
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_fraction(value: Optional[float], name: str) -> Optional[float]:
        """Stop/target oranlarini dogrular: None ya da (0, 1) araliginda olmalidir."""
        if value is None:
            return None
        v = float(value)
        if not 0.0 < v < 1.0:
            raise BacktestError(f"{name} 0 ile 1 arasinda bir oran olmalidir (orn. 0.05 = %5).")
        return v

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

        # --- Look-ahead duzeltmesi ---
        # next_open: i barinin kapanisinda olusan sinyali bir bar ileri kaydir,
        # boylece i+1 barinin ACILISINDAN islem yapilir (gelecegi gormeden).
        if self.execution == "next_open":
            buy_signals = buy_signals.shift(1, fill_value=False)
            sell_signals = sell_signals.shift(1, fill_value=False)
            exec_src = "Open" if "Open" in self.df.columns else "Close"
        else:  # "close" — eski davranis (sinyal bariyla ayni kapanis)
            exec_src = "Close"

        close = self.df["Close"].to_numpy()            # mark-to-market icin
        exec_prices = self.df[exec_src].to_numpy()      # islem gerceklestirme fiyati
        # Stop/target intrabar tetiklenir: bar ici en dusuk/en yuksek fiyat kullanilir.
        # OHLC yoksa kapanis fiyatina geri dusulur.
        high_arr = self.df["High"].to_numpy() if "High" in self.df.columns else close
        low_arr = self.df["Low"].to_numpy() if "Low" in self.df.columns else close
        dates = self.df.index
        buy_arr = buy_signals.to_numpy()
        sell_arr = sell_signals.to_numpy()

        cash = self.initial_balance
        shares = 0.0
        in_position = False
        entry_price = 0.0
        stop_level = 0.0      # giriste hesaplanir; stop_loss kapali ise kullanilmaz
        target_level = 0.0    # giriste hesaplanir; take_profit kapali ise kullanilmaz
        equity = np.empty(len(close), dtype=float)

        self.signals.clear()
        self.trades.clear()

        for i in range(len(close)):
            exec_price = exec_prices[i]   # alis/satis bu fiyattan gerceklesir
            mtm_price = close[i]          # portfoy degeri kapanistan isaretlenir

            # Once cikis, sonra giris — ayni bar iki islem olmaz.
            if in_position:
                exit_price = None
                reason = None
                # Oncelik: koruyucu emirler (intrabar), sonra kural tabanli satis.
                # Ayni bar hem stop hem target gorunurse temkinli olup stop'u sayariz.
                if self.stop_loss is not None and low_arr[i] <= stop_level:
                    exit_price, reason = stop_level, "stop_loss"
                elif self.take_profit is not None and high_arr[i] >= target_level:
                    exit_price, reason = target_level, "take_profit"
                elif sell_arr[i]:
                    exit_price, reason = exec_price, "signal"

                if exit_price is not None:
                    cash = shares * exit_price * (1.0 - self.commission)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                    self.trades.append({
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(float(exit_price), 4),
                        "pnl_pct": round(pnl_pct, 4),
                        "reason": reason,
                    })
                    self.signals.append({
                        "date": self._fmt_date(dates[i]),
                        "side": "SELL",
                        "price": round(float(exit_price), 4),
                    })
                    shares = 0.0
                    in_position = False

            elif buy_arr[i]:
                shares = (cash * (1.0 - self.commission)) / exec_price
                entry_price = exec_price
                stop_level = entry_price * (1.0 - self.stop_loss) if self.stop_loss else 0.0
                target_level = entry_price * (1.0 + self.take_profit) if self.take_profit else 0.0
                cash = 0.0
                in_position = True
                self.signals.append({
                    "date": self._fmt_date(dates[i]),
                    "side": "BUY",
                    "price": round(float(exec_price), 4),
                })

            # Bar sonu portfoy degeri (mark-to-market)
            equity[i] = cash + shares * mtm_price

        # Acik pozisyon kaldiysa son fiyattan kapat (gerceklesmis K/Z icin)
        if in_position:
            last_price = close[-1]
            cash = shares * last_price * (1.0 - self.commission)
            pnl_pct = (last_price - entry_price) / entry_price * 100.0
            self.trades.append({
                "entry_price": round(entry_price, 4),
                "exit_price": round(float(last_price), 4),
                "pnl_pct": round(pnl_pct, 4),
                "reason": "end_of_data",
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
        win_pnls = [t["pnl_pct"] for t in self.trades if t["pnl_pct"] > 0]
        loss_pnls = [t["pnl_pct"] for t in self.trades if t["pnl_pct"] < 0]
        wins = len(win_pnls)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        # Ortalama kazanc/kayip ve profit factor (kar/zarar buyukluk orani).
        avg_win = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0.0
        avg_loss = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
        gross_profit = sum(win_pnls)
        gross_loss = abs(sum(loss_pnls))
        if gross_loss > 0:
            profit_factor: Optional[float] = round(gross_profit / gross_loss, 2)
        else:
            # Zarar eden islem yoksa oran tanimsiz; None ile isaretlenir.
            profit_factor = None

        # Cikis nedenlerinin dagilimi (stop_loss/take_profit/signal/end_of_data).
        exit_reasons: Dict[str, int] = {}
        for t in self.trades:
            r = t.get("reason", "signal")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Maksimum dusus (peak-to-trough)
        eq = self.equity_curve
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100.0) if len(eq) else 0.0

        # Al-ve-tut kiyaslamasi: ilk gecerli kapanistan son kapanisa pasif getiri.
        close_series = self.df["Close"].dropna()
        if len(close_series) >= 2 and close_series.iloc[0] != 0:
            buy_hold_pct = (close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0] * 100.0
        else:
            buy_hold_pct = 0.0
        # Alpha: strateji getirisinin al-tut'a gore farki (artilik/eksiklik).
        alpha_pct = total_return_pct - buy_hold_pct

        # Sharpe orani: gunluk portfoy getirilerinden yillik (252 gun) olceklenir.
        returns = eq.pct_change().dropna()
        std = float(returns.std())
        if len(returns) > 1 and std > 0:
            sharpe = float(returns.mean() / std * np.sqrt(252))
        else:
            sharpe = 0.0

        return {
            "baslangic_bakiye": round(self.initial_balance, 2),
            "son_bakiye": round(final_equity, 2),
            "toplam_kar_zarar_pct": round(total_return_pct, 2),
            "al_tut_getiri_pct": round(buy_hold_pct, 2),
            "strateji_alpha_pct": round(alpha_pct, 2),
            "sharpe_orani": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "ort_kazanc_pct": round(avg_win, 2),
            "ort_kayip_pct": round(avg_loss, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "toplam_islem_sayisi": n_trades,
            "toplam_sinyal_sayisi": len(self.signals),
            "cikis_nedenleri": exit_reasons,
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
