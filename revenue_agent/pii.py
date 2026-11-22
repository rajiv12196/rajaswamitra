"""Mask personal identifiers before text reaches the language model.

Even with a local model, masking keeps identifiers out of prompts, logs and
the audit trail. Originals are kept only in memory for the officer's view.
"""
from __future__ import annotations

import re

PATTERNS = [
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("MOBILE", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
    ("BANK_AC", re.compile(r"(?i)(?:a/?c|account)(?:\s*(?:no\.?|number))?\s*[:#-]?\s*(\d{9,18})")),
]


def mask(text: str, existing: dict | None = None) -> tuple[str, dict]:
    """Return (masked_text, mapping placeholder -> original).
    `existing` lets new text reuse placeholders already assigned."""
    mapping: dict[str, str] = dict(existing or {})
    reverse: dict[str, str] = {v: k for k, v in mapping.items()}
    counters: dict[str, int] = {}
    for ph in mapping:
        kind, _, n = ph.strip("[]").rpartition("_")
        counters[kind] = max(counters.get(kind, 0), int(n))

    def placeholder(kind: str, value: str) -> str:
        if value in reverse:
            return reverse[value]
        counters[kind] = counters.get(kind, 0) + 1
        ph = f"[{kind}_{counters[kind]}]"
        mapping[ph] = value
        reverse[value] = ph
        return ph

    out = text
    for kind, pat in PATTERNS:
        if kind == "BANK_AC":
            out = pat.sub(lambda m: m.group(0).replace(m.group(1), placeholder(kind, m.group(1))), out)
        else:
            out = pat.sub(lambda m: placeholder(kind, m.group(0)), out)
    return out, mapping


def unmask(text: str, mapping: dict) -> str:
    for ph, val in mapping.items():
        text = text.replace(ph, val)
    return text
