"""
nlp_parser.py
-------------
Kullanicidan gelen Turkce dogal dil metinlerini, makinenin anlayabilecegi
yapilandirilmis JSON ticaret kurallarina ceviren NLP servisi.

Google Gemini API + Pydantic "Structured Output" kullanir.

Bagimliliklar:
    pip install google-genai python-dotenv pydantic

Ortam degiskeni (.env):
    GEMINI_API_KEY=...

Kullanim:
    from nlp_parser import BISTRuleParser
    parser = BISTRuleParser()
    rule = parser.parse("RSI 30'un altina dustugunde THYAO al.")
    print(rule.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "google-genai kurulu degil. Kurmak icin: pip install google-genai"
    ) from exc


# ---------------------------------------------------------------------- #
# Pydantic semasi (hedeflenen JSON yapisi)
# ---------------------------------------------------------------------- #
class Operator(str, Enum):
    """Bir kosulda kullanilabilecek karsilastirma operatorleri."""
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    EQUALS = "equals"


class Action(str, Enum):
    """Kurallarin tetikleyecegi islem."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Condition(BaseModel):
    """Tek bir teknik kosul."""
    indicator: str = Field(
        ...,
        description=(
            "Indikator adi. Gecerli degerler: price, open, high, low, RSI, STOCH_K, "
            "STOCH_D, SMA_20, SMA_50, SMA_100, SMA_200, EMA_12, EMA_26, EMA_50, EMA_200, "
            "MACD, MACD_Signal, MACD_Hist, ADX, BB_Upper, BB_Middle, BB_Lower, volume, Volume_SMA."
        ),
    )
    operator: Operator = Field(..., description="Karsilastirma operatoru.")
    value: Union[float, str] = Field(
        ...,
        description=(
            "Esik degeri. Sayisal esik ise sayi (orn. 30); baska bir seriye gore "
            "karsilastirma ise o indikatorun adi (orn. 'SMA_50', 'MACD_Signal', 'price')."
        ),
    )


class TradingRule(BaseModel):
    """Tam bir ticaret kurali (parse sonucu)."""
    asset: str = Field(..., description="Hisse/emtia sembolu, orn: THYAO.IS")
    conditions: List[Condition] = Field(
        ..., description="Islem icin saglanmasi gereken kosullar listesi."
    )
    action: Action = Field(..., description="BUY, SELL veya HOLD.")


# ---------------------------------------------------------------------- #
# System prompt
# ---------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are an expert financial rule-parsing assistant. Convert the user's natural-language
trading instruction into a structured rule matching the given JSON schema. The user may
write in ENGLISH or TURKISH (or mix them); understand both. Even if the text is terse or
colloquial, infer intent and map to the nearest valid indicator/operator.

English phrasing -> operator:
  "drops/falls below", "under", "less than"      -> less_than
  "rises/goes above", "over", "greater than"     -> greater_than
  "crosses above", "breaks above"                -> crosses_above
  "crosses below", "breaks below"                -> crosses_below
  "touches", "equals"                            -> equals
  "buy/long" -> BUY,  "sell/short/exit" -> SELL
English indicator words: "moving average"->SMA_50, "50-day moving average"->SMA_50,
"exponential/EMA"->EMA, "stochastic"->STOCH_K, "signal line"->MACD_Signal,
"bollinger upper/lower band"->BB_Upper/BB_Lower, "price/close"->price.

(Asagidaki Turkce kurallar da ayni sekilde gecerlidir.)

== 1) SEMBOL ==
- BIST hisseleri: sonuna ".IS" ekle. Ornek: THYAO->THYAO.IS, ASELS->ASELS.IS,
  GARAN->GARAN.IS, "Turk Hava Yollari"->THYAO.IS, "Garanti"->GARAN.IS, "Aselsan"->ASELS.IS.
- Emtialar (yfinance): Gumus->SI=F, Altin->GC=F, Ham petrol->CL=F, Bakir->HG=F,
  Dogalgaz->NG=F, Bitcoin->BTC-USD, Dolar/TL->TRY=X.

== 2) INDIKATORLER (yalnizca bu adlari kullan) ==
Fiyat:        price (kapanis), open, high, low
Momentum:     RSI, STOCH_K, STOCH_D
Hareketli ort: SMA_20, SMA_50, SMA_100, SMA_200, EMA_12, EMA_26, EMA_50, EMA_200
MACD:         MACD (cizgi), MACD_Signal (sinyal cizgisi), MACD_Hist (histogram)
Trend gucu:   ADX
Bollinger:    BB_Upper, BB_Middle, BB_Lower
Hacim:        volume, Volume_SMA (hacim ortalamasi)

Turkce -> indikator esleme ornekleri:
- "RSI", "goreli guc endeksi"                 -> RSI
- "50 gunluk ortalama/hareketli ortalama"     -> SMA_50
- "200 gunluk ortalama"                       -> SMA_200
- "20 gunluk ussel/EMA ortalama"              -> EMA_12 veya EMA_26 (en yakin)
- "stokastik"                                 -> STOCH_K
- "MACD sinyal cizgisini keserse"             -> MACD ve MACD_Signal
- "bollinger ust/alt bandi"                   -> BB_Upper / BB_Lower
- "hacim ortalamanin uzerine cikinca"         -> volume vs Volume_SMA
- "ADX 25 uzerindeyse (trend guclu)"          -> ADX
ONEMLI: Listede olmayan bir periyot istenirse (orn. "30 gunluk") EN YAKIN mevcut
periyodu sec (30 -> SMA_20 veya SMA_50). Asla listede olmayan bir ad uretme.

== 3) OPERATORLER (yalnizca bunlar) ==
less_than, greater_than, crosses_above, crosses_below, equals
- "altina duser/inerse/dusunce", "X'ten az/kucuk"     -> less_than
- "ustune cikar/gecince/asarsa", "X'ten fazla/buyuk"  -> greater_than
- "yukari keser/yukari kesince", "yukari kirar"        -> crosses_above
- "asagi keser/asagi kesince", "asagi kirar"           -> crosses_below
- "esit olunca/dokununca"                              -> equals

== 4) VALUE ALANI ==
- Sayisal esik ise SAYI yaz: RSI<30 -> value: 30.
- Baska bir seriye gore karsilastirma/kesisim ise o indikatorun ADINI string yaz:
  "fiyat 50 gunluk ortalamayi yukari keserse" -> {indicator:"price", operator:"crosses_above", value:"SMA_50"}
  "MACD sinyal cizgisini yukari keserse"      -> {indicator:"MACD", operator:"crosses_above", value:"MACD_Signal"}

== 5) COKLU KOSUL ==
Cumlede "ve", "ayrica", "hem ... hem" varsa her kosulu ayri bir condition olarak ekle
(hepsi AND ile birlesir). Ornek: "RSI 30 altinda VE fiyat SMA50 ustunde" -> 2 condition.

== 6) ISLEM (action) ==
"al/alim/long" -> BUY, "sat/satim/short/cik" -> SELL, belirsizse -> HOLD.

== 7) CIKTI ==
Yalnizca semaya uygun JSON uret; aciklama, yorum veya ekstra metin EKLEME.

ORNEK:
Girdi: "RSI 30'un altina dustugunde ve fiyat 50 gunluk ortalamanin uzerindeyse GARAN al."
Cikti: {"asset":"GARAN.IS","conditions":[
  {"indicator":"RSI","operator":"less_than","value":30},
  {"indicator":"price","operator":"greater_than","value":"SMA_50"}],"action":"BUY"}
"""


# ---------------------------------------------------------------------- #
# Parser sinifi
# ---------------------------------------------------------------------- #
class NLPParserError(Exception):
    """NLP ayristirma sirasinda olusan hata."""


# ====================================================================== #
# YEREL (KURAL TABANLI) COZUCU
# Gemini erisilemediginde (kota/ag yok) yaygin Turkce kaliplari regex ile
# cozer. Gemini kadar esnek degildir ama temel stratejileri yakalar.
# ====================================================================== #
_TR_MAP = str.maketrans("üışçöğÜİŞÇÖĞ", "uiscogUISCOG")


def _norm(text: str) -> str:
    """Turkce karakterleri sadelestirip kucuk harfe cevirir."""
    return text.translate(_TR_MAP).lower()


# Emtia / kripto / doviz adlari -> yfinance sembolu
_COMMODITY = {
    "gumus": "SI=F", "silver": "SI=F",
    "altin": "GC=F", "gold": "GC=F",
    "ham petrol": "CL=F", "petrol": "CL=F", "crude oil": "CL=F", "crude": "CL=F", "oil": "CL=F",
    "bakir": "HG=F", "copper": "HG=F",
    "dogalgaz": "NG=F", "natural gas": "NG=F",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "dolar": "TRY=X", "dollar": "TRY=X",
}

# Yaygin BIST hisse adlari -> sembol
_KNOWN_BIST = {
    "turk hava yollari": "THYAO.IS", "thy": "THYAO.IS", "thyao": "THYAO.IS",
    "garanti": "GARAN.IS", "garan": "GARAN.IS",
    "aselsan": "ASELS.IS", "asels": "ASELS.IS",
    "akbank": "AKBNK.IS", "akbnk": "AKBNK.IS",
    "is bankasi": "ISCTR.IS", "isbank": "ISCTR.IS", "isctr": "ISCTR.IS",
    "bim": "BIMAS.IS", "bimas": "BIMAS.IS",
    "eregli": "EREGL.IS", "eregl": "EREGL.IS",
    "tupras": "TUPRS.IS", "tuprs": "TUPRS.IS",
    "sasa": "SASA.IS",
    "koc holding": "KCHOL.IS", "kchol": "KCHOL.IS",
    "sabanci": "SAHOL.IS", "sahol": "SAHOL.IS",
    "ford": "FROTO.IS", "froto": "FROTO.IS",
    "pegasus": "PGSUS.IS", "pgsus": "PGSUS.IS",
    "sisecam": "SISE.IS", "sise": "SISE.IS",
}


def _detect_asset(raw: str, norm: str) -> Optional[str]:
    """Metinden hisse/emtia sembolu cikarir (kelime sinirlariyla)."""
    def _has(word: str) -> bool:
        # Kelime siniri: "altin" -> "altina" icinde ESLESMEZ.
        return re.search(r"\b" + re.escape(word) + r"\b", norm) is not None

    # 1) Emtia adlari (en uzun eslesme oncelikli)
    for name in sorted(_COMMODITY, key=len, reverse=True):
        if _has(name):
            return _COMMODITY[name]
    # 2) Bilinen BIST adlari
    for name in sorted(_KNOWN_BIST, key=len, reverse=True):
        if _has(name):
            return _KNOWN_BIST[name]
    # 3) Orijinal metinde 4-6 harfli buyuk harf token (orn. THYAO) -> .IS
    m = re.search(r"\b([A-ZÇĞİÖŞÜ]{4,6})\b", raw)
    if m:
        return m.group(1).translate(_TR_MAP).upper() + ".IS"
    return None


def _nearest(period: int, options: tuple) -> int:
    return min(options, key=lambda o: abs(o - period))


def _ma_column(c: str) -> Optional[str]:
    """Cumlecikteki bir hareketli ortalama referansini kolon adina cevirir."""
    m = re.search(r"\b(sma|ema)\s*[_-]?\s*(\d+)", c)  # "sma50", "ema_200", "ema 26"
    if m:
        period = int(m.group(2))
        if m.group(1) == "ema":
            return f"EMA_{_nearest(period, (12, 26, 50, 200))}"
        return f"SMA_{_nearest(period, (20, 50, 100, 200))}"
    if "ortalama" in c or "moving average" in c:
        pm = re.search(r"(\d+)\s*-?\s*(?:gun|day)", c)  # "50 gunluk", "200-day"
        is_ema = "ussel" in c or "exponential" in c or "ema" in c
        if is_ema:
            period = int(pm.group(1)) if pm else 12
            return f"EMA_{_nearest(period, (12, 26, 50, 200))}"
        period = int(pm.group(1)) if pm else 50
        return f"SMA_{_nearest(period, (20, 50, 100, 200))}"
    return None


def _detect_indicator(clause: str) -> Optional[str]:
    """Bir cumlecikten indikator adini cikarir (kolon adlandirmasiyla)."""
    c = clause
    if "rsi" in c:
        return "RSI"
    if "stokastik" in c or "stoch" in c:
        return "STOCH_K"
    if "adx" in c:
        return "ADX"

    has_price = "fiyat" in c or "kapanis" in c or "price" in c or "close" in c
    has_band = "bollinger" in c or "bant" in c or "band" in c
    has_signal = "sinyal" in c or "signal" in c
    ma = _ma_column(c)

    # MACD tek basina (fiyatla kiyas yoksa) MACD'dir.
    if "macd" in c and not has_price:
        return "MACD"
    # "fiyat ... ortalama/bant/sinyal" => ozne fiyat, digeri deger olur.
    if has_price and (ma or has_band or has_signal):
        return "price"
    if "macd" in c:
        return "MACD"
    if has_band:
        if "ust" in c or "upper" in c:
            return "BB_Upper"
        if "alt" in c or "lower" in c:
            return "BB_Lower"
        return "BB_Middle"
    if ma:
        return ma
    if has_price:
        return "price"
    return None


def _detect_operator(clause: str) -> Optional[str]:
    c = clause
    if "yukari kes" in c or "yukari kir" in c or "yukari gec" in c \
            or "crosses above" in c or "cross above" in c or "breaks above" in c:
        return "crosses_above"
    if "asagi kes" in c or "asagi kir" in c \
            or "crosses below" in c or "cross below" in c or "breaks below" in c:
        return "crosses_below"
    if "dokun" in c or "esit" in c or "touch" in c or "equal" in c:
        return "equals"
    if ("alt" in c or "asagi" in c or "dus" in c or "az" in c or "kucuk" in c or "inince" in c or "iner" in c
            or "below" in c or "under" in c or "less than" in c or "drop" in c or "fall" in c):
        return "less_than"
    if ("ust" in c or "uzer" in c or "yukar" in c or "gec" in c or "asar" in c or "fazla" in c or "buyuk" in c or "cik" in c
            or "above" in c or "over" in c or "greater" in c or "rise" in c or "exceed" in c):
        return "greater_than"
    return None


def _detect_value(clause: str, indicator: str):
    """Esik degerini bulur: once gercek bir sayi, yoksa baska bir seri referansi."""
    # Periyot/seri olarak gecen sayilari (deger sanmamak icin) disla:
    exclude = set(re.findall(r"\b(?:sma|ema)\s*[_-]?\s*(\d+)", clause))
    exclude |= set(re.findall(r"(\d+)\s*-?\s*(?:gun|day)", clause))

    for n in re.findall(r"\d+(?:[.,]\d+)?", clause):
        base = re.split(r"[.,]", n)[0]
        if base not in exclude:
            return float(n.replace(",", "."))

    # Sayisal esik yok: baska bir seriye gore karsilastirma referansi dondur
    ma = _ma_column(clause)
    if ma and ma != indicator:
        return ma
    if "sinyal" in clause or "signal" in clause:
        return "MACD_Signal"
    if "bollinger" in clause or "band" in clause or "bant" in clause:
        if "ust" in clause or "upper" in clause:
            return "BB_Upper"
        if "alt" in clause or "lower" in clause:
            return "BB_Lower"
    return None


def local_parse(text: str) -> TradingRule:
    """Gemini olmadan, regex ile temel bir TradingRule cikarir."""
    raw = text.strip()
    norm = _norm(raw)

    asset = _detect_asset(raw, norm)
    if not asset:
        raise NLPParserError(
            "Yerel cozucu metinden bir hisse/emtia cikaramadi. "
            "Sembolu acikca yazin (orn. THYAO) veya Gemini'yi etkinlestirin."
        )

    is_sell = re.search(r"\bsat\b|satim|sat\.|\bsell\b|\bshort\b|\bexit\b", norm) is not None
    action = "SELL" if is_sell else "BUY"

    # Cumleyi 've'/'and'/virgul ile cumleciklere bol, her birinden bir kosul cikar
    clauses = re.split(r"\bve\b|\band\b|,|;", norm)
    conditions = []
    for cl in clauses:
        ind = _detect_indicator(cl)
        if not ind:
            continue
        op = _detect_operator(cl)
        if not op:
            continue
        val = _detect_value(cl, ind)
        # "X gunluk ortalamanin altina/ustune" gibi kaliplari "price vs MA" olarak yorumla:
        # indikator bir hareketli ortalama ama esik degeri yoksa, ozne fiyattir.
        if val is None and ind.startswith(("SMA", "EMA")):
            val = ind
            ind = "price"
        if val is None:
            continue
        conditions.append({"indicator": ind, "operator": op, "value": val})

    if not conditions:
        raise NLPParserError(
            "Yerel cozucu metinden bir kosul cikaramadi. "
            "Daha acik yazin (orn. 'RSI 30 altina dusunce') veya Gemini'yi etkinlestirin."
        )

    return TradingRule(asset=asset, conditions=conditions, action=action)


class BISTRuleParser:
    """Turkce ticaret talimatlarini TradingRule nesnesine ceviren servis."""

    # Cozulen kurallarin diske kaydedildigi kalici onbellek dosyasi.
    _CACHE_PATH = Path(__file__).with_name("rule_cache.json")

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        # Anahtar yoksa LLM'siz (yalnizca yerel cozucu) modda calis.
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        # Bellek-ici + diskten yuklenen kalici onbellek.
        self._cache: dict[str, TradingRule] = self._load_cache()

    @property
    def has_llm(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ #
    # Kalici onbellek (disk)
    # ------------------------------------------------------------------ #
    def _load_cache(self) -> dict:
        try:
            data = json.loads(self._CACHE_PATH.read_text(encoding="utf-8"))
            return {k: TradingRule.model_validate(v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            data = {k: v.model_dump(mode="json") for k, v in self._cache.items()}
            self._CACHE_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass  # onbellek yazimi kritik degil; sessizce gec

    # ------------------------------------------------------------------ #
    # Gemini cagrisi
    # ------------------------------------------------------------------ #
    def _call_gemini(self, text: str) -> TradingRule:
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=TradingRule,
                        temperature=0.0,
                    ),
                )
                break
            except Exception as exc:
                msg = str(exc)
                transient = any(s in msg for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded"))
                if transient and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                    continue
                raise NLPParserError(f"Gemini API hatasi: {exc}") from exc

        rule = response.parsed
        if rule is None:
            try:
                rule = TradingRule.model_validate_json(response.text)
            except Exception as exc:
                raise NLPParserError(
                    f"Model ciktisi semaya uymuyor: {exc}\nHam cikti: {response.text}"
                ) from exc
        return rule

    def parse(self, text: str, use_cache: bool = True, prefer: str = "auto") -> TradingRule:
        """
        Turkce metni TradingRule'a cevirir.
        Oncelik:  kalici onbellek -> Gemini (varsa) -> yerel regex cozucu.
        prefer="local" verilirse Gemini hic denenmez (kota korunur).
        """
        if not text or not text.strip():
            raise NLPParserError("Bos metin ayristirilamaz.")

        key = text.strip().lower()
        if use_cache and key in self._cache:
            return self._cache[key]

        rule: Optional[TradingRule] = None

        # 1) Gemini (istenirse ve mumkunse)
        if prefer != "local" and self.has_llm:
            try:
                rule = self._call_gemini(text.strip())
            except NLPParserError:
                rule = None  # asagida yerel cozucuye dus

        # 2) Yerel regex cozucu (Gemini yoksa/basarisizsa)
        if rule is None:
            rule = local_parse(text)  # cikaramazsa NLPParserError firlatir

        # Onbellege yaz (bellek + disk)
        self._cache[key] = rule
        self._save_cache()
        return rule

    def parse_to_json(self, text: str, indent: int = 2) -> str:
        """parse() sonucunu JSON string olarak dondurur (diger modullere uygun)."""
        return self.parse(text).model_dump_json(indent=indent)


# ---------------------------------------------------------------------- #
# Test blogu
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    ornek = (
        "RSI 30'un altina dustugunde ve 50 gunluk hareketli ortalama "
        "yukari kesildiginde THYAO al."
    )

    try:
        parser = BISTRuleParser()
        kural = parser.parse(ornek)
        print("=== Girdi ===")
        print(ornek)
        print("\n=== Yapilandirilmis Kural (JSON) ===")
        print(kural.model_dump_json(indent=2))
    except NLPParserError as err:
        print(f"[HATA] {err}")
