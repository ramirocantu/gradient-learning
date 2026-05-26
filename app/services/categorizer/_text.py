"""Shared text-normalization helpers for the categorizer package."""

from __future__ import annotations

_TYPOGRAPHIC_TO_ASCII = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
    }
)


def normalize_typographic_punctuation(s: str) -> str:
    """Map Unicode curly quotes/apostrophes to ASCII equivalents.

    The AAMC outline JSON contains U+2019 in names like "Piaget's stages...".
    The LLM echoes paths back with U+0027 (straight apostrophe), causing
    exact-string lookups to fail. Normalize both sides to ASCII.
    Scope: exactly the four typographic-punctuation codepoints above.
    """
    return s.translate(_TYPOGRAPHIC_TO_ASCII)
