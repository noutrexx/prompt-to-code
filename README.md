# 📈 Prompt-to-Code · Strateji Backtest

Türkçe doğal dilde yazılan borsa stratejilerini otomatik olarak çalıştırılabilir
ticaret kurallarına çeviren, BIST 100 ve emtia verisiyle **geçmişe dönük test
(backtest)** yapan uçtan uca bir sistem.

> "RSI 35'in altına düştüğünde THYAO al" → yapay zeka kurala çevirir → veri
> çekilir → strateji test edilir → metrikler ve grafik döner.

---

## 🏗️ Mimari

```
Kullanıcı metni
      │
      ▼
nlp_parser.py     →  Gemini + Pydantic ile yapılandırılmış kural (TradingRule)
      │
      ▼
data_engine.py    →  yfinance verisi + RSI / MACD / SMA50 / SMA200
      │
      ▼
backtest_engine.py → komisyonlu simülasyon + metrikler + sinyaller
      │
      ▼
app.py (FastAPI)  →  /api/run-strategy  →  JSON (grafik + metrik)
      │
      ▼
frontend/         →  Lightweight Charts: mum grafik, AL/SAT okları, portföy eğrisi
```

## 📦 Modüller

| Dosya | Görev |
|-------|-------|
| `data_engine.py` | yfinance ile veri çekme + teknik indikatörler (TTL'li önbellek) |
| `nlp_parser.py` | Gemini ile Türkçe metin → `TradingRule` JSON (sonuç önbellekli) |
| `backtest_engine.py` | Kural setiyle vektörel backtest, metrik hesabı |
| `app.py` | FastAPI servisi + CORS + statik frontend |
| `frontend/index.html` | Modern arayüz, mum + portföy grafiği, metrik paneli |

## 🚀 Kurulum

```bash
pip install -r requirements.txt
```

`.env` dosyası oluştur (örnek için `.env.example`):

```
GEMINI_API_KEY=senin_anahtarin
```

> API anahtarını [Google AI Studio](https://aistudio.google.com/app/apikey)'dan alabilirsin.

## ▶️ Çalıştırma

```bash
python app.py
# veya:  uvicorn app:app --reload
```

Ardından tarayıcıda **http://127.0.0.1:8000/** adresini aç.

## 🔬 Metrikler

- **Toplam Kâr/Zarar (%)** — başlangıç 10.000 ₺, her işlemde %0.1 komisyon
- **Win Rate (%)** — kârlı işlemlerin oranı
- **Max Drawdown (%)** — zirveden en derin düşüş
- **Toplam İşlem Sayısı**

## 🧪 Modülleri tek tek test etme

```bash
python data_engine.py       # THYAO verisi + indikatörler
python nlp_parser.py        # örnek metni kurala çevirir (API anahtarı gerekir)
python backtest_engine.py   # örnek kural setiyle backtest
```

## ⚠️ Notlar

- Gemini ücretsiz katmanda `gemini-2.5-pro` kapalıdır; varsayılan model `gemini-2.5-flash`.
- `.env` asla commit edilmez (`.gitignore`'da).
- Bu proje eğitim/araştırma amaçlıdır, **yatırım tavsiyesi değildir**.

## 🗺️ Yol Haritası

- [ ] Kullanıcının ayrı giriş + çıkış kuralı yazabilmesi
- [ ] Bir sonraki bar açılışından giriş (look-ahead düzeltmesi)
- [ ] Çoklu sembol / strateji karşılaştırması
- [ ] Birim testleri (pytest)
