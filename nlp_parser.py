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

import os
import time
from enum import Enum
from typing import List, Union

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
Sen uzman bir finansal kural ayristirma (parsing) asistanisin. Gorevin, kullanicinin
Turkce dogal dilde yazdigi ticaret talimatini, verilen JSON semasina uygun
yapilandirilmis bir kurala cevirmektir. Kullanici eksik/gunluk konusma diliyle yazsa
bile niyetini cikar ve en yakin gecerli indikator/operatore esle.

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


class BISTRuleParser:
    """Turkce ticaret talimatlarini TradingRule nesnesine ceviren servis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise NLPParserError(
                "GEMINI_API_KEY bulunamadi. .env dosyasina ekleyin."
            )
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
        # Ayni metin tekrar gelirse Gemini'yi (ve kotayi) bos yere harcamamak icin onbellek.
        self._cache: dict[str, TradingRule] = {}

    def parse(self, text: str, use_cache: bool = True) -> TradingRule:
        """Verilen Turkce metni yapilandirilmis bir TradingRule'a cevirir."""
        if not text or not text.strip():
            raise NLPParserError("Bos metin ayristirilamaz.")

        key = text.strip().lower()
        if use_cache and key in self._cache:
            return self._cache[key]

        # Gecici hatalarda (503 yogunluk / 429 kota) ustel bekleme ile yeniden dene.
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=text.strip(),
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
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait)
                    continue
                raise NLPParserError(f"Gemini API hatasi: {exc}") from exc

        # google-genai, response_schema verildiginde .parsed icinde
        # dogrudan Pydantic nesnesini dondurur.
        rule = response.parsed
        if rule is None:
            # Yedek: ham metni Pydantic ile dogrula.
            try:
                rule = TradingRule.model_validate_json(response.text)
            except Exception as exc:
                raise NLPParserError(
                    f"Model ciktisi semaya uymuyor: {exc}\nHam cikti: {response.text}"
                ) from exc

        self._cache[key] = rule  # onbellege yaz
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
