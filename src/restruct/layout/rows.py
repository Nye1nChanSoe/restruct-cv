"""Visual rows: separately extracted cells that share a baseline.

A row preserves its left and right cells without deciding what they mean.
"""
from __future__ import annotations

import statistics as statistics_module
from typing import Any

from restruct.document.stats import DocumentStatistics
from restruct.document.types import ExtractedLine
from restruct.geometry import rounded, union


def _visual_rows(
    lines: list[ExtractedLine],
    line_indexes: range,
    statistics: DocumentStatistics,
) -> list[list[tuple[int, ExtractedLine]]]:
    """Cluster separately extracted left/right cells into visual rows.

    Grouped on typographic baselines rather than on box overlap. Box tops and
    bottoms move with descenders and with the tallest glyph present, so two
    cells of one row can have visibly different boxes while sharing a baseline
    exactly -- which is what makes them a row.
    """
    ordered = sorted(
        ((index, lines[index]) for index in line_indexes),
        key=lambda item: (item[1].page, item[1].bbox[1], item[1].bbox[0]),
    )
    tolerance = statistics.baseline_tolerance
    rows: list[list[tuple[int, ExtractedLine]]] = []
    for line_index, line in ordered:
        if rows and rows[-1][0][1].page == line.page and tolerance > 0:
            row_baseline = statistics_module.median(
                [item.baseline for _, item in rows[-1]]
            )
            if abs(line.baseline - row_baseline) <= tolerance:
                rows[-1].append((line_index, line))
                rows[-1].sort(key=lambda item: item[1].bbox[0])
                continue
        rows.append([(line_index, line)])
    return rows

def _row_value(row: list[tuple[int, ExtractedLine]]) -> dict[str, Any]:
    row_box = union(line.bbox for _, line in row)
    return {
        "page": row[0][1].page,
        "bbox": rounded(row_box),
        "detectionMethod": "geometry_row",
        "cells": [
            {
                "text": line.text,
                "page": line.page,
                "bbox": rounded(line.bbox),
                "lineIndex": line_index,
            }
            for line_index, line in row
        ],
    }
