"""Guards on the block continuation rule.

Nine copies of this arithmetic were spread across the section parsers and had
drifted into three different rules. These tests pin the one rule, and pin the
two axes on which the callers still deliberately differ.
"""

from __future__ import annotations

import pytest

from restruct.configs import SETTINGS
from restruct.layout.blocks import MINIMUM_GAP, continues_block, extend_block

LINE_HEIGHT = 10.0
MAXIMUM_GAP = LINE_HEIGHT * SETTINGS.section_router.paragraph_gap_multiplier


def line(top: float, left: float = 0.0, right: float = 100.0) -> tuple[float, ...]:
    return (left, top, right, top + LINE_HEIGHT)


def test_a_line_directly_below_continues_the_block() -> None:
    assert continues_block(
        line(0), line(LINE_HEIGHT), same_page=True, require_horizontal_overlap=True
    )


def test_a_slight_overlap_still_continues() -> None:
    """Glyph boxes include leading, so consecutive lines can overlap a little."""
    assert continues_block(
        line(0),
        line(LINE_HEIGHT + MINIMUM_GAP),
        same_page=True,
        require_horizontal_overlap=True,
    )


def test_a_large_gap_starts_a_new_block() -> None:
    assert not continues_block(
        line(0),
        line(LINE_HEIGHT + MAXIMUM_GAP + 1),
        same_page=True,
        require_horizontal_overlap=True,
    )


def test_a_line_above_never_continues() -> None:
    assert not continues_block(
        line(100), line(0), same_page=True, require_horizontal_overlap=True
    )


def test_a_different_page_never_continues() -> None:
    assert not continues_block(
        line(0), line(LINE_HEIGHT), same_page=False, require_horizontal_overlap=True
    )


def test_horizontal_overlap_separates_prose_from_hanging_bullets() -> None:
    """A column to the right is not a continuation; a hanging indent is."""
    indented = line(LINE_HEIGHT, left=200.0, right=300.0)
    assert not continues_block(
        line(0), indented, same_page=True, require_horizontal_overlap=True
    )
    assert continues_block(
        line(0), indented, same_page=True, require_horizontal_overlap=False
    )


def test_extend_block_joins_text_grows_the_box_and_keeps_evidence() -> None:
    block = {"text": "Perform preventive", "bbox": [0.0, 0.0, 50.0, 10.0]}
    extend_block(
        block,
        text="and corrective maintenance.",
        box=(0.0, 10.0, 80.0, 20.0),
        entities=[{"type": "url", "text": "example.com"}],
    )
    assert block["text"] == "Perform preventive\nand corrective maintenance."
    assert block["bbox"] == [0.0, 0.0, 80.0, 20.0]
    assert block["entities"] == [{"type": "url", "text": "example.com"}]


def test_extend_block_preserves_the_physical_line_break() -> None:
    """The newline stays so a later pass can still see where the line broke."""
    block = {"text": "first", "bbox": [0.0, 0.0, 10.0, 10.0]}
    extend_block(block, text="second", box=(0.0, 10.0, 10.0, 20.0))
    assert "\n" in block["text"]


def test_extend_block_adds_no_entities_key_when_there_is_no_evidence() -> None:
    block = {"text": "first", "bbox": [0.0, 0.0, 10.0, 10.0]}
    extend_block(block, text="second", box=(0.0, 10.0, 10.0, 20.0))
    assert "entities" not in block


def test_extend_block_appends_to_existing_evidence() -> None:
    block = {
        "text": "first",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "entities": [{"type": "url", "text": "a.com"}],
    }
    extend_block(
        block,
        text="second",
        box=(0.0, 10.0, 10.0, 20.0),
        entities=[{"type": "url", "text": "b.com"}],
    )
    assert [entity["text"] for entity in block["entities"]] == ["a.com", "b.com"]
