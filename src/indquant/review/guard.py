"""Prompt-injection defences for third-party text (news headlines) sent to the AI reviewer.

Defence in depth, kept deliberately small:
1. sanitise - Unicode-normalise, strip control / zero-width characters and markup, cap length;
2. scan     - flag text that looks like instructions aimed at an AI ("ignore previous
              instructions", "rate this BUY", fake role tags ...);
3. fence    - pass headlines inside a clearly delimited, numbered untrusted-data block;
4. authority - in code, the reviewer's output can only PASS / CAUTION / VETO a trade the
              quant model already proposed. It can never create, enlarge or re-price a
              trade, so a successful injection can at worst block a trade, never place one.
A detected injection attempt is itself treated as a red flag for that stock.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠﻿‪‫‬‭‮"))
_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier|the)\s+(instructions|prompts?|rules)",
    r"disregard\s+.{0,40}(instructions|rules|guidelines)",
    r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+instructions)",
    r"(system\s+prompt|developer\s+message|jailbreak)",
    r"^\s*(system|assistant|user)\s*:",
    r"</?\s*(system|instructions?|untrusted\w*|candidate|assistant)\b",
    r"(verdict|rating)\s*[:=]\s*(pass|buy|veto|caution)",
    r"(respond|reply|output|answer)\s+(only\s+)?(with|exactly)",
    r"(rate|mark|classify)\s+(this|the)\s+(stock|trade|company)\s+(as\s+)?(pass|buy|safe)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.MULTILINE)


def sanitize(text: str, max_len: int = 300) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[<>{}\[\]`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(unicodedata.normalize("NFKC", str(text or ""))))


def fence(items: list[dict]) -> tuple[str, list[int]]:
    """Numbered untrusted block; returns (text, ids of items that look like injections)."""
    lines, suspicious = [], []
    for k, item in enumerate(items, 1):
        title = sanitize(item.get("title", ""))
        if looks_like_injection(item.get("title", "")):
            suspicious.append(k)
        when = str(item.get("timestamp", ""))[:16]
        src = sanitize(item.get("source", ""), 40)
        lines.append(f"[{k}] {when} | {src} | {title}")
    return "<untrusted_headlines>\n" + "\n".join(lines) + "\n</untrusted_headlines>", suspicious
