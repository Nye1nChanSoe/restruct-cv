"""Bullet and numbered-list markers."""

from __future__ import annotations

import re

# U+F0B7 is the Private Use Area glyph Word exports for a Wingdings bullet,
# and U+200B / U+FEFF are zero-width characters PDF exporters leave after the
# marker. All three are invisible in an editor, so they are written as escapes
# on purpose: a literal is silently lost the moment anyone retypes the line.
BULLET_RE = re.compile(
    r"^\s*(?:[-+*\u2022\u25cf\u25aa\u25e6\u2023\uf0b7\u00a2]|\d+[.)])"
    r"[\s\u200b\ufeff]*"
)
