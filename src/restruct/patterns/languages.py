"""Language names.

Deterministic evidence that a ``Thai: Native`` line under a compound heading
belongs to the languages component rather than to the other half of it.

The list is deliberately a list. A language name is a closed class in a way
that a job title is not, so enumerating the ones that appear on resumes is
honest, whereas a pattern that tried to recognise "a language" by shape would
match every capitalised noun. Being incomplete is safe: an unrecognised
language means the block goes unclaimed, and unclaimed content is preserved
under ``others`` rather than misfiled.
"""

from __future__ import annotations

import re

# Ordered longest-first so "Chinese (Mandarin)" cannot be shadowed by a shorter
# alternative during matching.
LANGUAGE_NAMES = (
    "Arabic",
    "Bahasa Indonesia",
    "Bahasa Malaysia",
    "Bengali",
    "Burmese",
    "Cantonese",
    "Chinese",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Filipino",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Italian",
    "Japanese",
    "Javanese",
    "Kannada",
    "Khmer",
    "Korean",
    "Lao",
    "Malay",
    "Malayalam",
    "Mandarin",
    "Marathi",
    "Nepali",
    "Norwegian",
    "Persian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Sinhala",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Vietnamese",
)

# Accepts an optional parenthesised or dashed qualifier, because resumes write
# "Chinese (Mandarin)" and "English - British" as the label of one entry.
LANGUAGE_NAME_RE = re.compile(
    r"(?:" + "|".join(sorted(LANGUAGE_NAMES, key=len, reverse=True)) + r")"
    r"(?:\s*\([^)]{1,30}\))?",
    re.IGNORECASE,
)
