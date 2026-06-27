"""
compare_engine.py
-----------------
Ayni stratejiyi birden cok sembol uzerinde calistirip sonuclari kiyaslar
ve toplam getiriye gore siralar.

Veri cekme bagimliligi disaridan enjekte edilebilir (fetcher), boylece
agdan bagimsiz test edilebilir. Varsayilan fetcher data_engine'i kullanir.

Kullanim:
    from compare_engine import compare_symbols
    rows = compare_symbols(["THYAO.IS", "ASELS.IS"], buy_rule, sell_rule)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backtest_engine import Backtester, BacktestError


def _default_fetcher(symbol: str) -> pd.DataFrame:
    """Varsayilan veri kaynagi: yfinance + indikatorler (2 yil)."""
    # Gec import: ag/kutuphane bagimliligini yalnizca gercek kullanimda yukle.
    from data_engine import MarketDataFetcher

    return MarketDataFetcher(symbol, period="2y").run()


def compare_symbols(
    symbols: List[str],
    buy_rule: Dict[str, Any],
    sell_rule: Optional[Dict[str, Any]] = None,
    *,
    fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    **backtest_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Her sembol icin ayni kurali backtest eder ve sonuclari toplam getiriye gore
    azalan sirada dondurur. Bir sembol hata verirse digerlerini etkilemez;
    o satir ``ok=False`` ve ``error`` ile isaretlenir.

    Donen her satir: {symbol, ok, rank?, metrics?, error?}
    """
    if not symbols:
        raise ValueError("En az bir sembol verilmelidir.")

    fetch = fetcher or _default_fetcher
    # Ayni sembolun tekrarini koru ama sirayi bozma (ilk gorulus sirasi).
    seen: set[str] = set()
    unique_symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    results: List[Dict[str, Any]] = []
    for symbol in unique_symbols:
        try:
            df = fetch(symbol)
            # Kuralin 'asset' alanini ilgili sembole sabitle (kopya uzerinde).
            rule_for_symbol = {**buy_rule, "asset": symbol}
            bt = Backtester(df, rule_for_symbol, sell_rule, **backtest_kwargs)
            metrics = bt.run()
            results.append({"symbol": symbol, "ok": True, "metrics": metrics})
        except Exception as exc:  # bir sembolun hatasi tum kiyasi durdurmasin
            results.append({"symbol": symbol, "ok": False, "error": str(exc)})

    # Basarili sonuclari toplam getiriye gore sirala; hatalilar en sona.
    results.sort(
        key=lambda r: r["metrics"]["toplam_kar_zarar_pct"] if r["ok"] else float("-inf"),
        reverse=True,
    )
    rank = 1
    for row in results:
        if row["ok"]:
            row["rank"] = rank
            rank += 1
    return results
