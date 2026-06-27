<div align="center">

<img src="assets/banner.svg" alt="Prompt-to-Code" width="100%" />

<br/>

**Turn plain-language trading strategies into executable rules, then backtest them on BIST 100 and commodity data — end to end.**

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)

![Uvicorn](https://img.shields.io/badge/Uvicorn-2094f3?style=for-the-badge&logo=gunicorn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Lightweight Charts](https://img.shields.io/badge/TradingView_Charts-131722?style=for-the-badge&logo=tradingview&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

<a href="#overview">Overview</a> ·
<a href="#features">Features</a> ·
<a href="#tech-stack">Tech Stack</a> ·
<a href="#architecture">Architecture</a> ·
<a href="#installation">Installation</a> ·
<a href="#usage">Usage</a>

</div>

---

## Overview

Prompt-to-Code lets you write a trading idea the way you would say it — in **English or Turkish** —
and turns it into a structured, testable rule. A natural-language sentence such as
*"Buy THYAO when RSI drops below 35"* is parsed by an AI model into a JSON rule, historical
price data and technical indicators are fetched, a commission-aware backtest is run, and the
results are returned as metrics and chart-ready data.

> **"Buy THYAO when RSI drops below 35"** &nbsp;→&nbsp; AI parses the rule &nbsp;→&nbsp; data is fetched &nbsp;→&nbsp; strategy is backtested &nbsp;→&nbsp; metrics and charts are returned.

<div align="center">
  <img src="assets/preview.svg" alt="Application preview" width="100%" />
</div>

---

## Features

| | |
|---|---|
| **Natural language to rules** | Gemini + Pydantic convert a sentence into a structured JSON rule |
| **Bilingual input** | Understands both English and Turkish phrasing |
| **Local fallback parser** | A regex engine handles common patterns when the API is unavailable |
| **Persistent cache** | Parsed rules are stored on disk, so repeated strategies cost no quota |
| **20+ indicators** | RSI, Stochastic, SMA/EMA (four periods), MACD, ADX, Bollinger Bands, volume |
| **Realistic backtest** | 0.1% commission, 10,000 ₺ start, next-bar-open fills (no look-ahead) |
| **Risk management** | Optional stop-loss and take-profit, triggered intrabar at the price level |
| **Multi-symbol compare** | Run one strategy across several tickers and rank them by return |
| **Rich metrics** | Buy & hold benchmark, alpha, Sharpe, profit factor, exit-reason breakdown |
| **Modern UI** | TradingView candlesticks, BUY/SELL markers, equity-curve tab |
| **Smart symbols** | BIST tickers get `.IS` automatically; commodities map to yfinance symbols |

---

## Tech Stack

<div align="center">

| Layer | Technologies |
|-------|--------------|
| **Backend** | ![Python](https://img.shields.io/badge/Python_3.12-3776AB?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-2094f3?logo=gunicorn&logoColor=white) |
| **Data & Analysis** | ![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![yfinance](https://img.shields.io/badge/yfinance-6001D2?logo=yahoo&logoColor=white) ![ta](https://img.shields.io/badge/ta_(technical_analysis)-FF6F00?logoColor=white) |
| **Artificial Intelligence** | ![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?logo=googlegemini&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) ![Lightweight Charts](https://img.shields.io/badge/Lightweight_Charts-131722?logo=tradingview&logoColor=white) |

</div>

---

## Architecture

```mermaid
flowchart LR
    U([User: plain-language strategy]) --> NLP

    subgraph BE["FastAPI Backend"]
        direction TB
        NLP[nlp_parser<br/>Gemini + Pydantic<br/>+ local fallback] --> DATA[data_engine<br/>yfinance + indicators]
        DATA --> BT[backtest_engine<br/>simulation + metrics]
    end

    BT --> API[/api/run-strategy<br/>JSON: chart + metrics]
    API --> FE([Frontend<br/>candles, BUY/SELL, equity])

    CACHE[(rule_cache.json)] -.-> NLP
    NLP -.-> CACHE
```

**Flow:** text → rule (NLP) → data + indicators → commission-aware backtest → JSON → charts.

---

## Project Structure

```
prompt-to-code/
├── data_engine.py        # yfinance data + 20+ technical indicators (cached)
├── nlp_parser.py         # Gemini/regex: text -> TradingRule + persistent cache
├── backtest_engine.py    # backtest sim: fills, stop/target, metrics
├── compare_engine.py     # run one strategy across many symbols, ranked
├── app.py                # FastAPI service + CORS + static frontend
├── frontend/
│   └── index.html        # modern UI (Lightweight Charts)
├── tests/                # network-free pytest suite
├── assets/               # README images (SVG)
├── requirements.txt      # pinned runtime dependencies
├── requirements-dev.txt  # test dependencies (pytest, httpx)
└── .env.example          # environment variable template
```

---

## Installation

```bash
git clone https://github.com/noutrexx/prompt-to-code.git
cd prompt-to-code
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```env
GEMINI_API_KEY=your_api_key
CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
RATE_LIMIT_PER_MINUTE=10
MAX_CONCURRENT_STRATEGIES=2
MAX_REQUEST_BODY_BYTES=4096
```

> Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
> The app also runs **without** a key — in that case the local regex parser is used.

## Usage

```bash
python app.py        # or: uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** in your browser.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is network-free (synthetic OHLC data) and covers the backtest engine
and the API guards.

---

## API

### `POST /api/run-strategy`

This expensive endpoint is protected by:

- an allowlisted CORS origin list;
- a maximum 500-character strategy and 4 KB request body;
- a per-client request limit, defaulting to 10 requests per minute;
- a concurrent strategy execution limit, defaulting to 2.

Tune these values through environment variables before deployment. Requests over the rate
limit return `429`; requests above current execution capacity return `503`.

The built-in limiter is process-local. Multi-instance production deployments should also
enforce a shared rate limit and abuse protection at the API gateway or reverse proxy.

**Request:**
```json
{ "strateji_metni": "Buy THYAO when RSI drops below 35" }
```

**Response (excerpt):**
```json
{
  "asset": "THYAO.IS",
  "rule": { "conditions": [ { "indicator": "RSI", "operator": "less_than", "value": 35 } ], "action": "BUY" },
  "metrics": { "toplam_kar_zarar_pct": 39.34, "win_rate_pct": 80.0, "max_drawdown_pct": -21.07, "toplam_islem_sayisi": 5 },
  "signals": [ { "date": "2024-08-09", "side": "BUY", "price": 294.77 } ],
  "candles": [ ... ], "sma50": [ ... ], "sma200": [ ... ], "equity": [ ... ]
}
```

### `POST /api/compare-strategy`

Runs one strategy across multiple symbols (max 5) and returns them ranked by total return.

**Request:**
```json
{ "strateji_metni": "Buy when RSI drops below 35", "semboller": ["THYAO.IS", "ASELS.IS", "GARAN.IS"] }
```

**Response (excerpt):**
```json
{
  "rule": { "conditions": [ ... ], "action": "BUY" },
  "results": [
    { "symbol": "ASELS.IS", "ok": true, "rank": 1, "metrics": { "toplam_kar_zarar_pct": 41.2 } },
    { "symbol": "THYAO.IS", "ok": true, "rank": 2, "metrics": { "toplam_kar_zarar_pct": 18.7 } },
    { "symbol": "BADSYM",   "ok": false, "error": "veri bulunamadi" }
  ]
}
```

Failing symbols are isolated (`ok: false`) and sorted last, so one bad ticker never aborts the comparison.

### `GET /api/health`
Returns service status and NLP mode (Gemini or local fallback).

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Total P/L (%)** | 10,000 ₺ start, 0.1% commission per trade |
| **Buy & Hold (%)** | Passive return over the same period (benchmark) |
| **Alpha (%)** | Strategy return minus buy & hold — does it actually add value? |
| **Sharpe** | Annualized risk-adjusted return from the equity curve |
| **Win Rate (%)** | Share of profitable trades |
| **Profit Factor** | Gross profit ÷ gross loss (`null` when there are no losing trades) |
| **Avg Win / Loss (%)** | Mean return of winning and losing trades |
| **Max Drawdown (%)** | Deepest peak-to-trough decline |
| **Exit reasons** | Breakdown of exits: `signal` / `stop_loss` / `take_profit` / `end_of_data` |
| **Trades** | Number of completed buy/sell round trips |

> Trades execute at the **next bar's open** after a signal (no look-ahead bias).
> Optional **stop-loss / take-profit** levels are checked intrabar against the bar's high/low.

---

## Roadmap

- [ ] Separate user-defined entry and exit rules
- [x] Next-bar-open entry (look-ahead correction)
- [x] Stop-loss / take-profit risk management
- [x] Multi-symbol / multi-strategy comparison
- [x] Richer metrics (buy & hold, alpha, Sharpe, profit factor)
- [x] Unit tests (pytest)

---

## Disclaimer

This project is for **educational and research purposes only**. Past performance does not
guarantee future results, and nothing produced here constitutes **financial advice**.

<div align="center">
<br/>
<sub>Natural-language algorithmic trading · BIST 100 + Commodities</sub>
</div>
