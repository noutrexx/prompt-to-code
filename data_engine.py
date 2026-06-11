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
from typing import Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    def fetch(self) -> pd.DataFrame:
        """yfinance ile son 1 yillik gunluk veriyi ceker."""
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
        logger.info("'%s' icin %d satir veri cekildi.", self.symbol, len(df))
        return df

    # ------------------------------------------------------------------ #
    # 2) Teknik indikatorler
    # ------------------------------------------------------------------ #
    def add_indicators(self) -> pd.DataFrame:
        """RSI(14), MACD(12,26,9), SMA(50), SMA(200) kolonlarini ekler."""
        if self.df is None:
            raise MarketDataError("Once fetch() cagrilmalidir.")

        close = self.df["Close"]

        # RSI (14)
        self.df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

        # MACD (12, 26, 9)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        self.df["MACD"] = macd.macd()
        self.df["MACD_Signal"] = macd.macd_signal()
        self.df["MACD_Hist"] = macd.macd_diff()

        # SMA (50) ve SMA (200)
        self.df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
        self.df["SMA_200"] = SMAIndicator(close=close, window=200).sma_indicator()

        return self.df

    # ------------------------------------------------------------------ #
    # 3) Null temizleme
    # ------------------------------------------------------------------ #
    def clean(self) -> pd.DataFrame:
        """NaN iceren satirlari temizler."""
        if self.df is None:
            raise MarketDataError("Temizlenecek veri yok. Once fetch() cagrilmalidir.")
        before = len(self.df)
        self.df = self.df.dropna()
        logger.info("Temizleme: %d -> %d satir.", before, len(self.df))
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
