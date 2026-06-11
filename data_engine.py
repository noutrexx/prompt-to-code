"""
data_engine.py
---------------
BIST 100 hisseleri ve emtialar icin gecmis veri ceken ve teknik
indikatorleri hesaplayan bagimsiz bir servis.

Bagimliliklar:
    pip install yfinance ta pandas

Kullanim:
    fetcher = MarketDataFetcher("THYAO.IS")
    df = fetcher.run()              # indikatorlu temiz DataFrame
    payload = fetcher.to_json()    # JSON string
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Basit modul-seviye onbellek: (sembol, period, interval) -> (zaman, DataFrame)
# Gunluk veri sik degismedigi icin varsayilan TTL 1 saat.
_CACHE: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 3600


class MarketDataError(Exception):
    """Veri cekme / isleme sirasinda olusan anlamli hata."""


class MarketDataFetcher:
    """Tek bir sembol icin veri ceker, indikator ekler ve disa aktarir."""

    def __init__(self, symbol: str, period: str = "1y", interval: str = "1d") -> None:
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            raise MarketDataError("Gecerli bir sembol girilmelidir (ornek: 'THYAO.IS').")
        self.symbol = symbol.strip().upper()
        self.period = period
        self.interval = interval
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # 1) Veri cekme
    # ------------------------------------------------------------------ #
    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """yfinance ile gunluk veriyi ceker (TTL'li onbellek destegiyle)."""
        cache_key = (self.symbol, self.period, self.interval)

        # Onbellek kontrolu: taze kayit varsa yfinance'i hic cagirma.
        if use_cache and cache_key in _CACHE:
            ts, cached_df = _CACHE[cache_key]
            if time.time() - ts < _CACHE_TTL_SECONDS:
                self.df = cached_df.copy()
                logger.info("'%s' verisi onbellekten alindi (%d satir).", self.symbol, len(cached_df))
                return self.df

        try:
            df = yf.download(
                self.symbol,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # ag / kutuphane kaynakli hatalar
            raise MarketDataError(f"'{self.symbol}' verisi cekilemedi: {exc}") from exc

        if df is None or df.empty:
            raise MarketDataError(
                f"'{self.symbol}' icin veri bulunamadi. Sembol hatali olabilir."
            )

        # yfinance bazen MultiIndex kolon dondurur; tek seviyeye indir.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        self.df = df
        _CACHE[cache_key] = (time.time(), df.copy())  # onbellege yaz
        logger.info("'%s' icin %d satir veri cekildi.", self.symbol, len(df))
        return df

    # ------------------------------------------------------------------ #
    # 2) Teknik indikatorler
    # ------------------------------------------------------------------ #
    def add_indicators(self) -> pd.DataFrame:
        """
        Genis bir teknik indikator seti ekler:
        RSI, SMA (20/50/100/200), EMA (12/26/50/200), MACD,
        Bollinger Bantlari, Stochastic, ADX ve hacim ortalamasi.
        """
        if self.df is None:
            raise MarketDataError("Once fetch() cagrilmalidir.")

        df = self.df
        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

        # --- Momentum ---
        df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df["STOCH_K"] = stoch.stoch()          # %K
        df["STOCH_D"] = stoch.stoch_signal()   # %D

        # --- Trend: SMA / EMA ---
        for w in (20, 50, 100, 200):
            df[f"SMA_{w}"] = SMAIndicator(close=close, window=w).sma_indicator()
        for w in (12, 26, 50, 200):
            df[f"EMA_{w}"] = EMAIndicator(close=close, window=w).ema_indicator()

        # MACD (12, 26, 9)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        # ADX (trend gucu)
        df["ADX_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()

        # --- Volatilite: Bollinger Bantlari (20, 2) ---
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()

        # --- Hacim ---
        df["Volume_SMA_20"] = SMAIndicator(close=vol, window=20).sma_indicator()

        self.df = df
        return self.df

    # ------------------------------------------------------------------ #
    # 3) Null temizleme
    # ------------------------------------------------------------------ #
    def clean(self) -> pd.DataFrame:
        """
        Yalnizca fiyat (OHLC) verisi eksik satirlari atar.
        Indikator isinma (warm-up) NaN'lari KORUNUR; cunku backtest motoru
        kosul degerlendirirken NaN'i otomatik olarak 'saglanmadi' (False) sayar.
        Boylece sadece kisa indikator kullanan stratejiler tum veriyi kullanir.
        """
        if self.df is None:
            raise MarketDataError("Temizlenecek veri yok. Once fetch() cagrilmalidir.")
        before = len(self.df)
        self.df = self.df.dropna(subset=["Open", "High", "Low", "Close"])
        logger.info("Temizleme (sadece OHLC): %d -> %d satir.", before, len(self.df))
        return self.df

    # ------------------------------------------------------------------ #
    # Yardimci: tum akisi calistir
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """fetch -> add_indicators -> clean adimlarini sirayla calistirir."""
        self.fetch()
        self.add_indicators()
        self.clean()
        return self.df

    # ------------------------------------------------------------------ #
    # 4) JSON disa aktarim (wrapper)
    # ------------------------------------------------------------------ #
    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Veriyi JSON string olarak dondurur. Tarihler string olarak tutulur.
        Eger veri henuz islenmediyse otomatik olarak run() cagirir.
        """
        if self.df is None:
            self.run()

        out = self.df.copy()
        out.index = out.index.strftime("%Y-%m-%d")  # tarihleri string yap
        records = out.reset_index().to_dict(orient="records")

        payload = {
            "symbol": self.symbol,
            "period": self.period,
            "interval": self.interval,
            "rows": len(records),
            "data": records,
        }
        return json.dumps(payload, ensure_ascii=False, indent=indent, default=str)


# ---------------------------------------------------------------------- #
# Test blogu
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        fetcher = MarketDataFetcher("THYAO.IS")
        df = fetcher.run()

        print("\n=== Son 5 satir (indikatorlu) ===")
        print(df.tail())

        print("\n=== JSON ciktisi (ilk kayit) ===")
        payload = json.loads(fetcher.to_json())
        print(f"Sembol: {payload['symbol']} | Satir sayisi: {payload['rows']}")
        if payload["data"]:
            print(json.dumps(payload["data"][-1], ensure_ascii=False, indent=2, default=str))

    except MarketDataError as err:
        logger.error("Hata: %s", err)
