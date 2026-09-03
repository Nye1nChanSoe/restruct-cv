"""Guards on the shared box arithmetic.

Before geometry.py these expressions were copied across the parsers, so a fix in
one place silently left the others behind. Pinning them here keeps the single
copy honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
import pytest

from restruct.geometry import (
    estimate_span_box,
    horizontal_overlap,
    pixel_box,
    rounded,
    union,
    union_by_page,
    vertical_overlap,
)


@dataclass(frozen=True)
class FakeLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]


def test_rounded_matches_the_debug_artifact_precision() -> None:
    assert rounded((1.23456, 2.0, 3.99999, 4.5)) == [1.23, 2.0, 4.0, 4.5]


def test_rounded_accepts_a_rect() -> None:
    assert rounded(pymupdf.Rect(1.111, 2.222, 3.333, 4.444)) == [1.11, 2.22, 3.33, 4.44]


def test_pixel_box_scales_for_debug_renders() -> None:
    from restruct.configs import SETTINGS

    scale = SETTINGS.debug.scale
    assert pixel_box((10.0, 20.0, 30.0, 40.0)) == (
        round(10 * scale),
        round(20 * scale),
        round(30 * scale),
        round(40 * scale),
    )


def test_union_covers_every_box() -> None:
    result = union([(0, 0, 10, 5), (20, 3, 25, 9)])
    assert rounded(result) == [0.0, 0.0, 25.0, 9.0]


def test_union_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError):
        union([])


def test_union_by_page_keeps_pages_separate() -> None:
    lines = [
        FakeLine(1, "a", (0, 0, 10, 5)),
        FakeLine(1, "b", (2, 8, 30, 12)),
        FakeLine(2, "c", (1, 1, 4, 4)),
    ]
    assert union_by_page(lines) == [
        {"page": 1, "bbox": [0.0, 0.0, 30.0, 12.0]},
        {"page": 2, "bbox": [1.0, 1.0, 4.0, 4.0]},
    ]


def test_vertical_overlap_is_negative_when_boxes_are_apart() -> None:
    assert vertical_overlap((0, 0, 10, 5), (0, 3, 10, 9)) == 2
    assert vertical_overlap((0, 0, 10, 5), (0, 8, 10, 9)) < 0


def test_horizontal_overlap_clamps_at_zero() -> None:
    """Row grouping relies on the clamp; a negative width must read as no overlap."""
    assert horizontal_overlap((0, 0, 10, 5), (5, 0, 20, 5)) == 5
    assert horizontal_overlap((0, 0, 10, 5), (30, 0, 40, 5)) == 0


def test_estimate_span_box_locates_a_range_proportionally() -> None:
    line = FakeLine(1, "abcdefghij", (0.0, 0.0, 100.0, 10.0))
    box = estimate_span_box(line, 2, 5)
    assert box is not None
    assert rounded(box) == [20.0, 0.0, 50.0, 10.0]


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [("", 0, 1), ("abc", 2, 2), ("abc", -1, 2), ("abc", 1, 99)],
)
def test_estimate_span_box_refuses_impossible_ranges(text: str, start: int, end: int) -> None:
    """An out-of-range span falls back to the whole line rather than inventing a box."""
    assert estimate_span_box(FakeLine(1, text, (0, 0, 10, 10)), start, end) is None
