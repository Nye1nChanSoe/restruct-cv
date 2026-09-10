"""The zero-width characters exporters leave in extracted text.

These are written as ``\\uXXXX`` escapes and referred to by codepoint in prose,
never pasted as literals. A literal renders as nothing in every editor, so it
is silently lost the moment anyone retypes the line -- which has already
happened once in this package.

The set lived in three places before this module: the bullet pattern named it
as escapes, the metadata split named it again, and the heading cleaner pasted
the literals. Only one of the three also answered the question ingestion needed
-- whether a line is *nothing but* invisible characters -- and none did, which
is how a line holding a single U+200B became an empty experience record.

- **U+200B** ZERO WIDTH SPACE, left after a list marker by Google Docs and Word
  PDF exports. This is the one that appears in real resumes.
- **U+FEFF** ZERO WIDTH NO-BREAK SPACE, a byte-order mark stranded mid-stream.
- **U+200C**, **U+200D** the zero-width non-joiner and joiner, meaningful in
  Arabic and Indic scripts but not in the Latin text these strip.
- **U+00AD** SOFT HYPHEN, an invisible line-break hint.

Every one is a Unicode format character, so ``str.strip()`` removes none of
them and ``str.isspace()`` is false for all five.
"""

from __future__ import annotations

# Ordered by how often they actually turn up in extracted resume text.
ZERO_WIDTH_CHARACTERS = "\u200b\ufeff\u200c\u200d\u00ad"

# Everything is_blank() treats as carrying no content: the zero-width set plus
# the whitespace str.strip() would have removed on its own.
_BLANK_CHARACTERS = ZERO_WIDTH_CHARACTERS + "\t\n\r\v\f "


def without_invisibles(text: str) -> str:
    """``text`` with every zero-width character removed.

    Offsets into the result do not index the original, so this is for text
    being *compared* or *classified*, never for text a later stage will reverse
    onto its source line. Where offsets matter, strip at the edges instead.
    """
    for character in ZERO_WIDTH_CHARACTERS:
        text = text.replace(character, "")
    return text


def is_blank(text: str) -> bool:
    """Whether ``text`` carries no visible content.

    Asked at the ingestion boundary, where a line holding only a zero-width
    space has to count as blank. Testing ``strip()`` is not enough: it keeps
    such a line, which then reaches the parsers as content and opens a record
    that owns nothing.
    """
    return not text.strip(_BLANK_CHARACTERS)
