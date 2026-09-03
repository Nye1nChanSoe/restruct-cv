"""Visual rows: separately extracted cells that share a baseline.

A row preserves its left and right cells without deciding what they mean.
"""
from __future__ import annotations

from typing import Any

import pymupdf

from restruct.document.types import ExtractedLine
from restruct.geometry import rounded, union, vertical_overlap


def _visual_rows(
    lines: list[ExtractedLine],
    line_indexes: range,
) -> list[list[tuple[int, ExtractedLine]]]:
    """Cluster separately extracted left/right cells into visual rows."""
    ordered = sorted(
        ((index, lines[index]) for index in line_indexes),
        key=lambda item: (item[1].page, item[1].bbox[1], item[1].bbox[0]),
    )
    rows: list[list[tuple[int, ExtractedLine]]] = []
    for line_index, line in ordered:
        line_box = pymupdf.Rect(line.bbox)
        if rows and rows[-1][0][1].page == line.page:
            row_box = union(item.bbox for _, item in rows[-1])
            overlap = vertical_overlap(row_box, line_box)
            minimum_height = min(row_box.height, line_box.height)
            center_difference = abs(
                (row_box.y0 + row_box.y1) / 2 - (line_box.y0 + line_box.y1) / 2
            )
            if overlap >= minimum_height * 0.45 or center_difference <= minimum_height * 0.35:
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
