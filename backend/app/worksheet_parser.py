# backend/app/worksheet_parser.py
from __future__ import annotations
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback problems if PDF parsing fails
_FALLBACK = [
    {"dividend": 144, "divisor": 12},
    {"dividend": 96, "divisor": 8},
    {"dividend": 105, "divisor": 7},
    {"dividend": 247, "divisor": 6},
    {"dividend": 336, "divisor": 4},
    {"dividend": 189, "divisor": 9},
]


def parse_worksheet(pdf_path: str | Path) -> list[dict]:
    """Extract long division problems from a worksheet PDF.

    Returns a list of {"dividend": int, "divisor": int} dicts, ordered
    left-to-right, top-to-bottom as they appear on the page.
    Falls back to _FALLBACK if parsing fails or nothing is found.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("pdfplumber not installed — using fallback problems")
        return list(_FALLBACK)

    problems: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                problems.extend(_extract_from_text(text))
    except Exception as exc:
        logger.warning("PDF parse error (%s) — using fallback", exc)
        return list(_FALLBACK)

    if not problems:
        logger.warning("No problems found in PDF — using fallback")
        return list(_FALLBACK)

    logger.info("Parsed %d problems from worksheet", len(problems))
    return problems


def _extract_from_text(text: str) -> list[dict]:
    """Parse problems from extracted PDF text using multiple heuristics."""
    problems: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(dividend: int, divisor: int) -> None:
        if divisor > 0 and dividend >= divisor and (dividend, divisor) not in seen:
            seen.add((dividend, divisor))
            problems.append({"dividend": dividend, "divisor": divisor})

    # Pattern 1: "144 ÷ 12" or "144÷12"
    for m in re.finditer(r'(\d+)\s*[÷]\s*(\d+)', text):
        add(int(m.group(1)), int(m.group(2)))

    # Pattern 2: long division bracket format lines like "12 ) 144" or "12)144"
    # Common in worksheet generators
    for m in re.finditer(r'(\d+)\s*\)\s*(\d+)', text):
        divisor, dividend = int(m.group(1)), int(m.group(2))
        add(dividend, divisor)

    # Pattern 3: fraction-style "144 / 12" (only if no ÷ problems found)
    if not problems:
        for m in re.finditer(r'(\d+)\s*/\s*(\d+)', text):
            add(int(m.group(1)), int(m.group(2)))

    return problems
