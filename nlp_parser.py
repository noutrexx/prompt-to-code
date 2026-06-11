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
        description="Indikator adi: RSI, SMA_50, SMA_200, MACD, price vb.",
    )
    operator: Operator = Field(..., description="Karsilastirma operatoru.")
    value: Union[float, str] = Field(
        ...,
        description="Esik degeri. Sayi (orn. 30) veya 'price' gibi bir referans olabilir.",
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
Sen bir finansal kural ayristirma (parsing) asistanisin. Gorevin, kullanicinin
Turkce dogal dilde yazdigi ticaret talimatini, verilen JSON semasina uygun
yapilandirilmis bir kurala cevirmektir.

KESIN KURALLAR:
1. BIST 100 hisseleri istendiginde sembolun sonuna mutlaka ".IS" ekle.
   Ornek: "THYAO" -> "THYAO.IS", "ASELS" -> "ASELS.IS".
   Emtialar icin yfinance sembollerini kullan: Gumus -> "SI=F", Altin -> "GC=F",
   Ham petrol -> "CL=F".
2. Indikatorleri standart adlandirmayla yaz:
   - RSI            -> "RSI"
   - 50 gunluk SMA  -> "SMA_50"
   - 200 gunluk SMA -> "SMA_200"
   - MACD           -> "MACD"
   - Fiyatin kendisi -> "price"
3. Operatorler yalnizca sunlar olabilir:
   less_than, greater_than, crosses_above, crosses_below, equals.
   - "altina duser/dusunce"        -> less_than
   - "ustune cikar/gecince"        -> greater_than
   - "yukari keser/yukari kesince" -> crosses_above
   - "asagi keser/asagi kesince"   -> crosses_below
4. "value" sayisal bir esikse sayi olarak (orn. 30), baska bir seriye gore
   kesisim ise "price" gibi bir referans string olarak yazilir.
5. "al" -> BUY, "sat" -> SELL, belirsizse -> HOLD.
6. Yalnizca semaya uygun ciktiyi uret; aciklama veya ekstra metin EKLEME.
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

    def parse(self, text: str) -> TradingRule:
        """Verilen Turkce metni yapilandirilmis bir TradingRule'a cevirir."""
        if not text or not text.strip():
            raise NLPParserError("Bos metin ayristirilamaz.")

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
        except Exception as exc:
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
