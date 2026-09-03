"""Splitting a metadata line while preserving offsets into its source.

Every candidate keeps the character range it came from, so a later stage can
reverse a split that turns out to be wrong.
"""
from __future__ import annotations

import re


from restruct.patterns.education import DEGREE_RE, INSTITUTION_RE


def _metadata_candidates(
    text: str,
    date_spans: list[re.Match[str]],
    separator_re: re.Pattern[str],
    *,
    preserve_education_hyphens: bool = False,
) -> list[tuple[str, int, int]]:
    """Split metadata while retaining exact character offsets into its source line."""
    segment_ranges: list[tuple[int, int]] = []
    cursor = 0
    for separator in separator_re.finditer(text):
        if any(
            match.start() < separator.end() and separator.start() < match.end()
            for match in date_spans
        ):
            continue
        separator_text = separator.group(0).strip()
        if preserve_education_hyphens and separator_text == "-":
            left = text[cursor:separator.start()].strip()
            right = text[separator.end():].strip()
            should_split = bool(
                (DEGREE_RE.search(left) and INSTITUTION_RE.search(right))
                or INSTITUTION_RE.search(left)
            )
            if not should_split:
                continue
        segment_ranges.append((cursor, separator.start()))
        cursor = separator.end()
    segment_ranges.append((cursor, len(text)))

    candidates: list[tuple[str, int, int]] = []
    for base_start, base_end in segment_ranges:
        residual_ranges = [(base_start, base_end)]
        for date_match in date_spans:
            occupied_start, occupied_end = date_match.start(), date_match.end()
            next_ranges: list[tuple[int, int]] = []
            for start, end in residual_ranges:
                if occupied_end <= start or end <= occupied_start:
                    next_ranges.append((start, end))
                    continue
                if start < occupied_start:
                    next_ranges.append((start, occupied_start))
                if occupied_end < end:
                    next_ranges.append((occupied_end, end))
            residual_ranges = next_ranges
        for raw_start, raw_end in residual_ranges:
            raw = text[raw_start:raw_end]
            value = raw.strip(" \t,;:-–—\u200b\ufeff")
            if not value:
                continue
            start = raw_start + raw.find(value)
            candidates.append((value, start, start + len(value)))
    return candidates
