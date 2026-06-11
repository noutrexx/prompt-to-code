"""
app.py
------
Prompt-to-Code stratejisi icin FastAPI sunucusu.

Zincir:  strateji_metni -> nlp_parser (kural) -> data_engine (veri)
         -> backtest_engine (simulasyon) -> JSON (metrikler + grafik verisi)

Calistirma:
    pip install -r requirements.txt
    uvicorn app:app --reload
    # veya:  python app.py

Frontend, http://127.0.0.1:8000/ adresinden de servis edilir.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backtest_engine import Backtester, BacktestError
from data_engine import MarketDataError, MarketDataFetcher
from nlp_parser import BISTRuleParser, NLPParserError

# ---------------------------------------------------------------------- #
# Uygulama + CORS
# ---------------------------------------------------------------------- #
app = FastAPI(title="Prompt-to-Code Strateji API", version="1.0.0")

# CORS: frontend ayri bir origin'den (veya file://) cagirabilsin diye acik.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NLP parser'i tek sefer kur (her istekte yeniden olusturmamak icin).
# Anahtar yoksa parser yerel (regex) cozucu moduna gecer, yine de calisir.
_parser = BISTRuleParser()


# ---------------------------------------------------------------------- #
# Istek / yanit modelleri
# ---------------------------------------------------------------------- #
class StrategyRequest(BaseModel):
    strateji_metni: str = Field(..., description="Turkce strateji cumlesi.")


# ---------------------------------------------------------------------- #
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------- #
def _build_candles(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """DataFrame'i Lightweight Charts formatina cevirir (candlestick + SMA)."""
    candles, sma50, sma200 = [], [], []
    for ts, row in df.iterrows():
        t = pd.Timestamp(ts).strftime("%Y-%m-%d")
        candles.append({
            "time": t,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
        })
        if "SMA_50" in row and pd.notna(row["SMA_50"]):
            sma50.append({"time": t, "value": round(float(row["SMA_50"]), 4)})
        if "SMA_200" in row and pd.notna(row["SMA_200"]):
            sma200.append({"time": t, "value": round(float(row["SMA_200"]), 4)})
    return {"candles": candles, "sma50": sma50, "sma200": sma200}


def _build_equity(equity_curve) -> List[Dict[str, Any]]:
    """Backtester equity Series'ini Lightweight Charts line formatina cevirir."""
    if equity_curve is None:
        return []
    out = []
    for ts, val in equity_curve.items():
        out.append({
            "time": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "value": round(float(val), 2),
        })
    return out


def _derive_exit_rule(buy_rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tek bir AL kuralindan makul bir SAT (cikis) kurali turetir.
    - RSI gibi sayisal esikler aynalanir (orn. <35 -> >65).
    - Diger operatorler ters cevrilir (crosses_above <-> crosses_below).
    """
    inverse_op = {
        "less_than": "greater_than",
        "greater_than": "less_than",
        "crosses_above": "crosses_below",
        "crosses_below": "crosses_above",
        "equals": "equals",
    }
    exit_conditions = []
    for cond in buy_rule.get("conditions", []):
        value = cond["value"]
        indicator = str(cond["indicator"]).upper()
        # RSI / Stochastic 0-100 arasinda osilator oldugundan esik aynalanir (orn. 35 -> 65).
        if (indicator.startswith("RSI") or indicator.startswith("STOCH")) and isinstance(value, (int, float)):
            value = 100 - value
        exit_conditions.append({
            "indicator": cond["indicator"],
            "operator": inverse_op.get(cond["operator"], cond["operator"]),
            "value": value,
        })
    return {
        "asset": buy_rule.get("asset"),
        "conditions": exit_conditions,
        "action": "SELL",
    }


# ---------------------------------------------------------------------- #
# Endpoint'ler
# ---------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    # llm=True: Gemini aktif | llm=False: yalnizca yerel regex cozucu
    return {"status": "ok", "nlp_ready": True, "llm": _parser.has_llm}


@app.post("/api/run-strategy")
def run_strategy(req: StrategyRequest) -> Dict[str, Any]:
    """
    Türkçe strateji metnini uctan uca calistirir ve grafik + metrik dondurur.
    """
    text = (req.strateji_metni or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="strateji_metni bos olamaz.")

    # 1) Metni kurala cevir (NLP: Gemini varsa onu, yoksa yerel cozucuyu kullanir)
    try:
        rule = _parser.parse(text)
        # mode="json": enum'lari ("BUY", "less_than" vb.) string degerlerine cevirir.
        rule_dict = rule.model_dump(mode="json")
    except NLPParserError as exc:
        raise HTTPException(status_code=502, detail=f"NLP hatasi: {exc}")

    asset = rule_dict.get("asset")
    if not asset:
        raise HTTPException(status_code=422, detail="Metinden hisse/emtia cikarilamadi.")

    # 2) Veriyi cek (indikatorlerle)
    try:
        df = MarketDataFetcher(asset, period="2y").run()
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=f"Veri hatasi: {exc}")

    # 3) Backtest: kural AL ise giris, turetilen kural cikis olur.
    buy_rule = rule_dict
    sell_rule = _derive_exit_rule(rule_dict)
    try:
        bt = Backtester(df, buy_rule, sell_rule, initial_balance=10_000, commission=0.001)
        metrics = bt.run()
    except BacktestError as exc:
        raise HTTPException(status_code=500, detail=f"Backtest hatasi: {exc}")

    # 4) Grafik verisini hazirla ve dondur
    chart = _build_candles(df)
    return {
        "asset": asset,
        "rule": rule_dict,
        "exit_rule": sell_rule,
        "metrics": metrics,
        "signals": bt.signals,   # [{date, side, price}]
        "candles": chart["candles"],
        "sma50": chart["sma50"],
        "sma200": chart["sma200"],
        "equity": _build_equity(bt.equity_curve),  # portfoy degeri egrisi
    }


# ---------------------------------------------------------------------- #
# Frontend'i statik olarak servis et (kolaylik icin)
# ---------------------------------------------------------------------- #
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
