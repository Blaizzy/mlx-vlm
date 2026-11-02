# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""Strict readings of written values; no arithmetic, unit conversion or invented date components."""

from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from decimal import Decimal
from functools import cache

# Month names a date is written with (accents folded; Polish and Russian in the genitive a date uses), and weekdays,
# which a date may lead with.
_MONTHS = (
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
)
MONTHS = {
    name: month for month, names in enumerate(_MONTHS, 1) for name in names.split()
}
_ABBREVIATIONS = {}
for _name, _month in MONTHS.items():
    _ABBREVIATIONS.setdefault(_name[:3], set()).add(_month)
MONTHS.update(
    {
        name: next(iter(months))
        for name, months in _ABBREVIATIONS.items()
        if len(months) == 1
    }
)
MONTHS.update(janv=1, fevr=2, juil=7, sept=9)
WEEKDAYS = set(
    """monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun
lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag lunes
martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca
quarta quinta sexta""".split()
)
CJK_DATE = re.compile(
    r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]"
)  # 2026年10月20日, 2026년 10월 20일
NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)")
_MONTH_NAME = "|".join(sorted(MONTHS, key=len, reverse=True))
WORD_DATE = re.compile(
    rf"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{_MONTH_NAME})[ ,.]+(?:de\s+)?\d{{4}}"
    rf"|(?:{_MONTH_NAME})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b"
)  # in folded text: 20 octobre 2026

# Currencies by the decimal mark their amounts are written with: "" for none (whole units), None for both in use.
CURRENCIES = {
    ".": (
        "$",
        "£",
        "₹",
        "USD",
        "GBP",
        "CAD",
        "AUD",
        "NZD",
        "INR",
        "Rs.",
        "Rs",
        "MXN",
        "CNY",
        "RMB",
        "元",
        "HKD",
        "SGD",
    ),
    ",": ("R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY"),
    "": ("¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP"),
    None: ("€", "EUR", "CHF"),
}
_NAMES = {c.casefold(): c for cs in CURRENCIES.values() for c in cs}
_MARK = {c.casefold(): mark for mark, cs in CURRENCIES.items() for c in cs}
_CURRENCY = (
    "(?i:"
    + "|".join(
        (
            rf"\b{re.escape(c)}(?![a-z])"
            if c[0].isascii() and c[0].isalpha()
            else re.escape(c)
        )
        for c in sorted(_NAMES.values(), key=len, reverse=True)
    )
    + ")"
)
# Scale words, written apart or attached: unambiguous, so no context needed (German "Billion" is 10^12: left out).
SCALES = {
    **dict.fromkeys(("thousand", "mil", "tausend", "тыс", "千"), 10**3),
    **dict.fromkeys(("万", "萬"), 10**4),
    **dict.fromkeys(("lakh", "lakhs"), 10**5),
    **dict.fromkeys(("crore", "crores"), 10**7),
    **dict.fromkeys(
        (
            "million",
            "millions",
            "mio",
            "mln",
            "millón",
            "millones",
            "milhão",
            "milhões",
            "milione",
            "milioni",
            "млн",
        ),
        10**6,
    ),
    **dict.fromkeys(("亿", "億"), 10**8),
    **dict.fromkeys(
        (
            "billion",
            "billions",
            "milliard",
            "milliards",
            "milliarde",
            "milliarden",
            "mrd",
            "miliardo",
            "miliardi",
            "млрд",
        ),
        10**9,
    ),
    "trillion": 10**12,
}
_SCALE = re.compile(
    r"\s*("
    + "|".join(sorted(map(re.escape, SCALES), key=len, reverse=True))
    + r")\.?$",
    re.I,
)
_SUFFIX = re.compile(
    r"(?<=\d)(k|m|bn|b)$", re.I
)  # attached k, M, B: a magnitude only beside money or a count
_COUNTED = re.compile(
    r"\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|"
    r"downloads)\b",
    re.I,
)
_GROUPED = re.compile(
    r"(?<![\d.,])\d{1,3}(?:([,. \u00a0\u202f'’])\d{3})+(?!\1)([.,])\d+(?![\d.,])"
)  # 1 234,50
_SHORT = re.compile(
    r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])"
)  # 12,50: a decimal, never a group
_REPEATED = re.compile(
    r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])"
)  # 1.234.567: a group, never a decimal
_LOCALE = re.compile(
    r'\b(?:locale|language|lang)["\']?\s*[=:]\s*["\']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)'
    r"(?:[-_][a-z]{2})?\b"
)


def _one(values: set):
    """The one value evidence settles on; None when it holds none or disagrees with itself."""
    return next(iter(values)) if len(values) == 1 else None


def written_unit(text: str) -> str | None:
    """Only an explicitly written percent or currency marker; a currency sign stays a sign."""
    if text.rstrip().endswith("%"):
        return "%"
    markers = re.findall(_CURRENCY, text)
    return _NAMES[markers[0].casefold()] if len(markers) == 1 else None


def _currency_marks(text: str) -> set[str]:
    """The decimal marks of the currencies `text` writes ("" for whole units); a currency of both says nothing."""
    return {_MARK[c.casefold()] for c in re.findall(_CURRENCY, text)} - {None}


def _decimal_mark(written: str, context: str) -> str | None:
    """The decimal mark ("" when amounts have none), from the first evidence that speaks: the span's own currency,
    the document's other numbers, the document's currencies, a declared locale. Evidence that disagrees with itself
    decides nothing; a markup declaration's numbers (`<?xml version="1.0"?>`) are not the document's.
    """
    context = re.sub(r"<\?[^>]*\?>", "", context)
    numbers = {m[2] for m in _GROUPED.finditer(context)} | {
        m[1] for m in _SHORT.finditer(context)
    }
    numbers |= {"," if m[1] == "." else "." for m in _REPEATED.finditer(context)}
    locales = {
        "." if lang in ("en", "ja", "zh", "ko") else ","
        for lang in _LOCALE.findall(context.casefold())
    }
    return _one(
        next(
            (
                marks
                for marks in (
                    _currency_marks(written),
                    numbers,
                    _currency_marks(context),
                    locales,
                )
                if marks
            ),
            set(),
        )
    )


def parse_number(
    text: str, context: str = "", *, unit: str | None = None
) -> int | float | None:
    """The written magnitude, percent points included: its currency and percent sign set apart, its scale word
    applied; no implicit unit and no conversion. None when the writing does not settle one value.
    """
    context += f"\n{unit}" if unit and re.fullmatch(_CURRENCY, unit.strip()) else ""
    written = text
    text = (
        text.strip()
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .removesuffix("%")
        .rstrip()
    )
    negative = text.startswith("(") and text.endswith(")")
    text = text[1:-1].strip() if negative else text
    sign = currency = None
    for _ in range(2):  # one currency and one sign, in either order
        if currency is None and (
            m := re.fullmatch(rf"({_CURRENCY})\s*(.*)|(.*?)\s*({_CURRENCY})", text)
        ):
            currency, text = m[1] or m[4], (m[2] if m[1] else m[3]).strip()
        if sign is None and text[:1] in ("-", "+"):
            sign, text = text[0], text[1:].strip()
    if negative and sign:
        return None
    sign = sign or ""
    scale = 1
    if word := _SCALE.search(text):
        scale, text = SCALES[word[1].lower()] if word[1] != "Billion" else 0, text[
            : word.start()
        ]
    elif suffix := _SUFFIX.search(text):
        if not (
            re.search(_CURRENCY, written + "\n" + context) or _COUNTED.search(context)
        ):
            return None
        scale, text = {"k": 10**3, "m": 10**6, "b": 10**9, "bn": 10**9}[
            suffix[1].lower()
        ], text[: suffix.start()]
    digits = _digits(text, written, context)
    if digits is None or not scale:
        return None
    value = Decimal(("-" if negative else sign) + digits) * scale
    if value == value.to_integral_value():
        return int(value)
    result = float(value)
    return result if math.isfinite(result) else None


def _digits(text: str, written: str, context: str) -> str | None:
    """`text`'s number as plain digits and a point: the decimal mark from the evidence, the other mark a complete
    grouping (threes, or India's lakh grouping beside a rupee)."""
    if not re.fullmatch(r"\d[\d., '’]*", text):
        return None
    if re.search(r"[ '’]", text):  # a space or an apostrophe only groups
        grouping = re.fullmatch(r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?", text)
        return (
            None
            if grouping is None
            else _digits(
                re.sub(r"[ '’]", "", grouping[1]) + (grouping[2] or ""),
                written,
                context,
            )
        )
    if re.fullmatch(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?", text) and re.search(
        r"₹|\bINR\b|\bRs\b|lakh|crore", written + "\n" + context, re.I
    ):
        return text.replace(",", "")
    if "." in text and "," in text:
        point, group = (".", ",") if text.rfind(".") > text.rfind(",") else (",", ".")
        if not re.fullmatch(
            rf"\d{{1,3}}(?:{re.escape(group)}\d{{3}})+{re.escape(point)}\d+", text
        ):
            return None
        return text.replace(group, "").replace(point, ".")
    if "." not in text and "," not in text:
        return text
    mark = "." if "." in text else ","
    parts = text.split(mark)
    if not all(parts):
        return None
    point = _decimal_mark(written, context)
    if (
        len(parts) > 2 or point is not None and point != mark
    ):  # the mark groups: complete threes
        return (
            "".join(parts)
            if 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:])
            else None
        )
    if (
        point is None and len(parts[1]) == 3 and len(parts[0]) <= 3
    ):  # 1,250: a group or a decimal, unsettled
        return None
    return ".".join(parts)


def _fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )


# a declared numeric date format, in English, French (jj/mm/aaaa), German (tt.mm.jjjj), Spanish or Italian (gg/mm/aaaa)
_DAY, _MONTH, _YEAR = (
    r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)",
    r"(?:mm?|month|mois|monat|mes|mese)",
    r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)",
)
DATE_FORMATS = (
    (rf"\b{_DAY}\s*[/.-]\s*{_MONTH}\s*[/.-]\s*{_YEAR}\b", "dmy"),
    (rf"\b{_MONTH}\s*[/.-]\s*{_DAY}\s*[/.-]\s*{_YEAR}\b", "mdy"),
)


def date_order(context: str, folded: str | None = None) -> str | None:
    """Declared format/locale first, then the state's other unambiguous numeric dates; conflicts stay unresolved.
    `folded`: `context` already folded."""
    folded = _fold(context) if folded is None else folded
    explicit = set()
    for pattern, order in DATE_FORMATS:
        if re.search(pattern, folded):
            explicit.add(order)
    if explicit:
        return _one(explicit)
    locales = re.findall(
        r'\b(?:locale|language|lang|langue|sprache|idioma|lingua)["\']?\s*[=:]\s*["\']?'
        r"(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b",
        folded,
    )
    if locales:
        return _one(
            {
                "mdy" if locale.replace("_", "-") in ("en", "en-us") else "dmy"
                for locale in locales
            }
        )
    orders = set()
    for match in NUMERIC_DATE.finditer(context):
        a, b, year = int(match[1]), int(match[3]), int(match[4])
        if len(match[4]) != 4 or a > 31 or b > 31 or a < 1 or b < 1:
            continue
        for day, month, order in ((a, b, "dmy"), (b, a, "mdy")):
            try:
                dt.date(year, month, day)
            except ValueError:
                continue
            if (a > 12) != (b > 12):
                orders.add(order)
    return _one(orders)


def _candidates(
    year: int, a: int, b: int, order: str | None
) -> list[tuple[int, int, int]]:
    """(year, month, day) for `a`/`b`/year: day first, month first, or the one the context's `order` says."""
    dmy, mdy = (year, b, a), (year, a, b)
    return [dmy, mdy] if order is None else [dmy if order == "dmy" else mdy]


def parse_date(text: str, context: str = "") -> tuple[str | None, bool]:
    """ISO value and unresolved-ambiguity flag; an invalid or unrecognised span is (None, False)."""
    folded = cache(
        lambda: _fold(context)
    )  # the context folded and its date order, once, when a date needs them
    return _parse_date(
        text, context, folded, cache(lambda: date_order(context, folded()))
    )


def _parse_date(text: str, context: str, folded, order) -> tuple[str | None, bool]:
    text = text.strip()
    if not text:
        return None, False
    # Keep an explicitly written time and timezone, with no default time or timezone.
    timed = re.fullmatch(
        r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)", text
    )
    if timed:
        date, ambiguous = _parse_date(timed[1], context, folded, order)
        if date is None:
            return None, ambiguous
        try:
            clock = timed[2]
            if clock[1] == ":":
                clock = "0" + clock
            return dt.datetime.fromisoformat(date + "T" + clock).isoformat(), False
        except ValueError:
            return None, False
    numeric = NUMERIC_DATE.fullmatch(text)
    if numeric:
        a, b, c = int(numeric[1]), int(numeric[3]), int(numeric[4])
        if len(numeric[1]) == 4:
            candidates = [(a, b, c)]
        elif len(numeric[4]) == 4:
            candidates = _candidates(c, a, b, order())
        else:
            # A century is never invented: another unambiguous complete date must establish it.
            if len(numeric[4]) != 2:
                return None, False
            centuries = set()
            complete = [
                match[0]
                for match in NUMERIC_DATE.finditer(context)
                if len(match[1]) == 4 or len(match[4]) == 4
            ]
            complete.extend(match[0] for match in WORD_DATE.finditer(folded()))
            for candidate in complete:
                value, ambiguous = _parse_date(candidate, context, folded, order)
                if value is not None and not ambiguous:
                    centuries.add(int(value[:4]) // 100 * 100)
            if len(centuries) != 1:
                return None, False
            candidates = _candidates(centuries.pop() + c, a, b, order())
        values = set()
        for year, month, day in candidates:
            try:
                values.add(dt.date(year, month, day).isoformat())
            except ValueError:
                pass
        if len(values) > 1:
            return None, True
        return next(iter(values), None), False
    if cjk := CJK_DATE.fullmatch(text):
        try:
            return dt.date(*map(int, cjk.groups())).isoformat(), False
        except ValueError:
            return None, False
    words = _fold(text)
    words = re.sub(r"(?<=\d)(?:st|nd|rd|th|er)\b", "", words)
    words = re.sub(r"(?<=\d)\.?(?:o|a|°)\b", "", words)
    words = re.sub(r"\b(?:de|of|the|le|el|il|den)\b", " ", words)
    words = re.sub(r"[,.]", " ", words).split()
    if (
        words and words[0].removesuffix("-feira") in WEEKDAYS
    ):  # Tue, 20 Oct 2026; segunda-feira, 20 de ...
        words = words[1:]
    if len(words) != 3:
        return None, False
    if words[0] in MONTHS:
        month, day, year = MONTHS[words[0]], words[1], words[2]
    elif words[1] in MONTHS:
        day, month, year = words[0], MONTHS[words[1]], words[2]
    else:
        return None, False
    if not day.isdigit() or not year.isdigit() or len(year) != 4:
        return None, False
    try:
        return dt.date(int(year), month, int(day)).isoformat(), False
    except ValueError:
        return None, False
