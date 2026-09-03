"""Document-relative statistics.

The prototype decided layout questions with fixed numbers -- a 1.12x font-size
multiplier, a 1.25x line-gap multiplier, a 2.0pt overlap tolerance. Those work
on the resumes they were tuned against and fail quietly on documents set at a
different size or leading.

This module measures the same quantities from the document itself, so a
threshold reads as "larger than this document's body text" rather than "larger
than 1.12 times something". Nothing here interprets meaning; it only measures.

Every statistic degrades to a stated fallback on a document too sparse to
measure, rather than raising or returning a misleading zero.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, replace

from restruct.document.physical import BBox, Document, Page, Rule, TextLine
from restruct.geometry import free_vertical_bands

# A line must have some real text before it can inform typography statistics.
_MINIMUM_MEANINGFUL_CHARACTERS = 2

# Two sizes within this many points are the same size for grouping purposes;
# PDF exporters routinely emit 10.98 and 11.0 for the same run of text.
_SIZE_TOLERANCE = 0.2

# A line repeated on at least this share of pages is page furniture.
_REPEAT_PAGE_RATIO = 0.6

# Fraction of page height within which a repeated line counts as a header/footer.
_MARGIN_BAND = 0.15

# A column gutter must run clear for at least this share of a page's text
# height, and the text either side of it must co-occur vertically for at least
# this share of it. On the fixtures a real two-column page scores 0.96-1.00 and
# 0.69-0.83; the most column-like single-column page scores 0.37 and 0.28, so
# both gates sit in open space rather than being fitted to either population.
_GUTTER_BAND_RATIO = 0.5
_GUTTER_CO_OCCURRENCE_RATIO = 0.5

# Fewer lines than this on either side is a stray box, not a column.
_MINIMUM_COLUMN_LINES = 4

# A page with less text than this cannot be judged; a title page has no columns.
_MINIMUM_COLUMN_PAGE_LINES = 8

# Running headers usually carry the page number, so the literal text differs on
# every page. Comparing with digit runs folded away is what lets
# "Synthetic resume | Page 1" and "... | Page 3" recognise each other.
_DIGIT_RUN_RE = re.compile(r"\d+")


def _furniture_key(text: str) -> str:
    return _DIGIT_RUN_RE.sub("#", " ".join(text.split())).strip()


def _median(values: list[float], fallback: float = 0.0) -> float:
    return statistics.median(values) if values else fallback


@dataclass(frozen=True)
class Gutter:
    """A vertical corridor of a page that no line of text crosses.

    Its presence means the page has independent columns, and therefore that
    reading top-to-bottom does not recover the author's reading order.
    """

    page: int
    left: float
    right: float
    top: float
    bottom: float
    left_column_lines: int
    right_column_lines: int

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def bbox(self) -> BBox:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class PageStatistics:
    """Per-page geometry. Margins can differ between pages of one document."""

    number: int
    left_margin: float
    right_margin: float
    median_line_height: float
    median_line_gap: float


@dataclass(frozen=True)
class DocumentStatistics:
    """What this document's own typography and geometry look like."""

    body_font_size: float
    font_sizes: tuple[tuple[float, int], ...]
    bold_ratio: float
    median_character_width: float
    # Measured from real space glyphs. 0.0 when the document positions words by
    # advance alone and emits no spaces, which some exporters do.
    median_space_width: float
    median_line_height: float
    median_line_gap: float
    left_margin: float
    right_margin: float
    indentation_levels: tuple[float, ...]
    # Stored with digit runs folded to "#"; compare via is_page_furniture().
    repeated_headers: frozenset[str] = frozenset()
    repeated_footers: frozenset[str] = frozenset()
    horizontal_rules: tuple[Rule, ...] = ()
    # Empty on the single-column documents v1 targets; see detect_unsupported().
    column_gutters: tuple[Gutter, ...] = ()
    pages: tuple[PageStatistics, ...] = field(default=(), repr=False)

    # -- typography -------------------------------------------------------

    def is_larger_than_body(self, size: float, *, ratio: float = 1.0) -> bool:
        """Whether a size stands out against this document's body text."""
        if self.body_font_size <= 0:
            return False
        return size >= self.body_font_size * ratio + _SIZE_TOLERANCE

    def is_body_size(self, size: float) -> bool:
        return abs(size - self.body_font_size) <= _SIZE_TOLERANCE

    @property
    def mostly_bold(self) -> bool:
        """A document whose body is bold cannot use boldness as a signal."""
        return self.bold_ratio > 0.5

    # -- vertical rhythm --------------------------------------------------

    def is_paragraph_gap(self, gap: float) -> bool:
        """Whether a vertical gap is small enough to continue a block.

        Measured against this document's own leading rather than as a multiple
        of the two boxes involved, so an oversized line no longer licenses an
        oversized gap.

        Line gaps are strongly bimodal -- on these fixtures around 2.3pt within
        a paragraph against 14-16pt between blocks -- and this threshold sits
        between the two modes. Both terms are needed: twice the typical leading
        tracks tightly set documents, while half a line height keeps the rule
        sane where leading is near zero.
        """
        return gap <= max(self.median_line_gap * 2.0, self.median_line_height * 0.5)

    @property
    def word_gap_threshold(self) -> float:
        """The gap at which two glyphs stop being one word.

        Derived from this document's own space glyphs where it has them, and
        from character width otherwise. Never a fixed constant: the same
        absolute gap means different things at 8pt and at 14pt.
        """
        if self.median_space_width > 0:
            return self.median_space_width * 0.6
        return self.median_character_width * 0.3

    @property
    def cell_gap_threshold(self) -> float:
        """The gap at which a line is divided into cells rather than spaced.

        Set well clear of both populations rather than fitted between them: on
        the fixtures, ordinary word gaps reach 2.7x a space while real column
        gaps run 54-86x, so anything in that range separates them and a
        conservative multiple stays safe on justified text, where spaces
        legitimately stretch.
        """
        reference = max(self.median_space_width, self.median_character_width)
        return reference * 5.0 if reference > 0 else 0.0

    @property
    def baseline_tolerance(self) -> float:
        """How far two baselines may differ and still be one row.

        Half a line height: two genuinely separate lines sit a full line height
        plus leading apart, so anything closer than half of one cannot be a
        different line, whatever their font sizes.
        """
        if self.median_line_height > 0:
            return self.median_line_height * 0.5
        return self.body_font_size * 0.5

    # -- horizontal rhythm ------------------------------------------------

    def indentation_level(self, left: float) -> int:
        """Which measured indentation level a left edge belongs to."""
        for index, level in enumerate(self.indentation_levels):
            if abs(left - level) <= max(2.0, self.median_character_width):
                return index
        return len(self.indentation_levels)

    def separated_by_a_gutter(self, page: int, first: BBox, second: BBox) -> bool:
        """Whether two boxes sit in different columns of a multi-column page.

        Row grouping asks this before joining two boxes that share a baseline:
        on a page with independent columns, sharing a baseline is a coincidence
        of layout rather than evidence that the two belong to one row.
        """
        nearer, further = sorted((first, second), key=lambda box: box[0])
        return any(
            gutter.page == page
            and nearer[2] <= gutter.left
            and further[0] >= gutter.right
            for gutter in self.column_gutters
        )

    def is_page_furniture(
        self,
        text: str,
        *,
        top: float,
        bottom: float,
        page_height: float,
    ) -> bool:
        """Whether a line is a running header or footer.

        Text and position are both required. After digit runs are folded, a
        repeated page number and a bare year in a table cell are the same
        string, so text alone cannot tell them apart -- only sitting in a page
        margin does.
        """
        if page_height <= 0:
            return False
        key = _furniture_key(text)
        if key in self.repeated_headers and top <= page_height * _MARGIN_BAND:
            return True
        return (
            key in self.repeated_footers
            and bottom >= page_height * (1 - _MARGIN_BAND)
        )


def _column_lines(page: Page, statistics: "DocumentStatistics") -> list[TextLine]:
    """The lines on a page that can bound a column.

    Running headers and footers are excluded because they legitimately span
    every column at once: leaving the page footer in would let one line hide a
    gutter that runs the whole height of the page above it.
    """
    return [
        line
        for line in page.lines
        if line.text.strip()
        and line.is_horizontal
        and not statistics.is_page_furniture(
            line.text,
            top=line.bbox[1],
            bottom=line.bbox[3],
            page_height=page.height,
        )
    ]


def _gutter_at(
    x: float,
    lines: list[TextLine],
    top: float,
    bottom: float,
) -> tuple[float, float, int, int] | None:
    """The tallest clear band at ``x``, if it looks like a column boundary.

    Returns the band and how many lines sit each side of it, or None when the
    vertical corridor at ``x`` is too short, too empty, or has text on only one
    side of it.
    """
    height = bottom - top
    if height <= 0:
        return None

    crossing = [
        (line.bbox[1], line.bbox[3]) for line in lines if line.bbox[0] < x < line.bbox[2]
    ]
    best: tuple[float, float, int, int] | None = None
    for band_top, band_bottom in free_vertical_bands(crossing, top, bottom):
        if band_bottom - band_top < height * _GUTTER_BAND_RATIO:
            continue
        left_lines = [
            line
            for line in lines
            if line.bbox[2] <= x and line.bbox[3] > band_top and line.bbox[1] < band_bottom
        ]
        right_lines = [
            line
            for line in lines
            if line.bbox[0] >= x and line.bbox[3] > band_top and line.bbox[1] < band_bottom
        ]
        if len(left_lines) < _MINIMUM_COLUMN_LINES or len(right_lines) < _MINIMUM_COLUMN_LINES:
            continue

        # Two columns must be beside each other, not one above the other: a
        # page whose text is left-hand at the top and right-hand at the bottom
        # has a clear corridor between them and is still read top to bottom.
        left_extent = (min(l.bbox[1] for l in left_lines), max(l.bbox[3] for l in left_lines))
        right_extent = (min(l.bbox[1] for l in right_lines), max(l.bbox[3] for l in right_lines))
        co_occurrence = min(left_extent[1], right_extent[1]) - max(left_extent[0], right_extent[0])
        if co_occurrence < height * _GUTTER_CO_OCCURRENCE_RATIO:
            continue

        candidate = (band_top, band_bottom, len(left_lines), len(right_lines))
        if best is None or band_bottom - band_top > best[1] - best[0]:
            best = candidate
    return best


def _page_gutters(page: Page, statistics: "DocumentStatistics") -> list[Gutter]:
    """Every column gutter on one page, widest corridor first."""
    lines = _column_lines(page, statistics)
    if len(lines) < _MINIMUM_COLUMN_PAGE_LINES:
        return []

    left = min(line.bbox[0] for line in lines)
    right = max(line.bbox[2] for line in lines)
    top = min(line.bbox[1] for line in lines)
    bottom = max(line.bbox[3] for line in lines)

    # Scanned at a point per unit rather than at line edges: a gutter is a
    # region, and its width is what says whether it is a column boundary or the
    # ragged right edge of a paragraph.
    found: list[tuple[float, tuple[float, float, int, int]]] = []
    for step in range(math.ceil(left) + 1, math.floor(right)):
        band = _gutter_at(float(step), lines, top, bottom)
        if band is not None:
            found.append((float(step), band))

    gutters: list[Gutter] = []
    run: list[tuple[float, tuple[float, float, int, int]]] = []
    for entry in found + [None]:  # type: ignore[list-item]
        if entry is not None and (not run or entry[0] - run[-1][0] <= 1.0):
            run.append(entry)
            continue
        if run:
            # The tallest corridor in the run describes it; the run's extent is
            # the gutter's width.
            tallest = max(run, key=lambda item: item[1][1] - item[1][0])
            gutters.append(
                Gutter(
                    page=page.number,
                    left=run[0][0],
                    right=run[-1][0],
                    top=tallest[1][0],
                    bottom=tallest[1][1],
                    left_column_lines=tallest[1][2],
                    right_column_lines=tallest[1][3],
                )
            )
        run = [entry] if entry is not None else []

    minimum_width = max(2.0, statistics.median_character_width * 2)
    return sorted(
        (gutter for gutter in gutters if gutter.width >= minimum_width),
        key=lambda gutter: -gutter.width,
    )


def _column_gutters(document: Document, statistics: "DocumentStatistics") -> tuple[Gutter, ...]:
    return tuple(
        gutter for page in document.pages for gutter in _page_gutters(page, statistics)
    )


def _line_is_measurable(line: TextLine) -> bool:
    return (
        line.is_horizontal
        and sum(character.isalnum() for character in line.text)
        >= _MINIMUM_MEANINGFUL_CHARACTERS
    )


def _body_font_size(lines: list[TextLine]) -> float:
    """The size carrying the most text, not the most lines.

    Weighting by character count matters: a resume can have more heading lines
    than body lines while the body still holds most of the words.
    """
    weighted: Counter[float] = Counter()
    for line in lines:
        for span in line.spans:
            if span.size > 0:
                weighted[round(span.size, 1)] += len(span.text)
    if not weighted:
        return 0.0
    return weighted.most_common(1)[0][0]


def _repeated_lines(document: Document) -> tuple[frozenset[str], frozenset[str]]:
    """Lines appearing near the top or bottom of most pages.

    This replaces the fixed 'page N' regex, which only caught footers that
    happened to end in a number.
    """
    if len(document.pages) < 2:
        return frozenset(), frozenset()

    # Pages per key, not occurrences: a running footer appears once on each of
    # several pages, whereas five table cells on one page are not a footer no
    # matter how alike they look.
    headers: dict[str, set[int]] = {}
    footers: dict[str, set[int]] = {}
    for page in document.pages:
        if page.height <= 0:
            continue
        top_band = page.height * _MARGIN_BAND
        bottom_band = page.height * (1 - _MARGIN_BAND)
        for line in page.lines:
            stripped = line.text.strip()
            if not stripped:
                continue
            key = _furniture_key(stripped)
            if line.bbox[1] <= top_band:
                headers.setdefault(key, set()).add(page.number)
            elif line.bbox[3] >= bottom_band:
                footers.setdefault(key, set()).add(page.number)

    threshold = max(2, int(len(document.pages) * _REPEAT_PAGE_RATIO))
    return (
        frozenset(key for key, pages in headers.items() if len(pages) >= threshold),
        frozenset(key for key, pages in footers.items() if len(pages) >= threshold),
    )


def _indentation_levels(lines: list[TextLine], character_width: float) -> tuple[float, ...]:
    """Cluster left edges into the distinct indents the document actually uses."""
    tolerance = max(2.0, character_width)
    levels: list[float] = []
    for left in sorted(line.bbox[0] for line in lines):
        if not levels or left - levels[-1] > tolerance:
            levels.append(left)
    return tuple(round(level, 2) for level in levels)


def _page_statistics(page_number: int, lines: list[TextLine]) -> PageStatistics:
    heights = [line.height for line in lines if line.height > 0]
    gaps = [
        following.bbox[1] - current.bbox[3]
        for current, following in zip(lines, lines[1:])
        if following.bbox[1] >= current.bbox[1]
    ]
    return PageStatistics(
        number=page_number,
        left_margin=min((line.bbox[0] for line in lines), default=0.0),
        right_margin=max((line.bbox[2] for line in lines), default=0.0),
        median_line_height=_median(heights),
        # Negative gaps come from overlapping boxes and would drag the median.
        median_line_gap=_median([gap for gap in gaps if gap >= 0]),
    )


def measure(document: Document) -> DocumentStatistics:
    """Measure one document. Safe on an empty or unparseable document."""
    measurable = [line for line in document.lines if _line_is_measurable(line)]

    sizes: Counter[float] = Counter()
    bold_characters = total_characters = 0
    character_widths: list[float] = []
    for line in measurable:
        for span in line.spans:
            if span.size > 0:
                sizes[round(span.size, 1)] += 1
            length = len(span.text)
            total_characters += length
            if span.bold:
                bold_characters += length
            if span.character_width > 0:
                character_widths.append(span.character_width)

    character_width = _median(character_widths)
    space_widths = [
        token.width
        for line in measurable
        for span in line.spans
        for token in span.tokens
        if token.text.isspace() and token.width > 0
    ]
    page_statistics = tuple(
        _page_statistics(page.number, [l for l in page.lines if _line_is_measurable(l)])
        for page in document.pages
    )
    repeated_headers, repeated_footers = _repeated_lines(document)

    statistics = DocumentStatistics(
        body_font_size=_body_font_size(measurable),
        font_sizes=tuple(sorted(sizes.items(), key=lambda item: -item[1])),
        bold_ratio=bold_characters / total_characters if total_characters else 0.0,
        median_character_width=character_width,
        median_space_width=_median(space_widths),
        median_line_height=_median([p.median_line_height for p in page_statistics]),
        median_line_gap=_median([p.median_line_gap for p in page_statistics if p.median_line_gap > 0]),
        left_margin=min((p.left_margin for p in page_statistics if p.left_margin > 0), default=0.0),
        right_margin=max((p.right_margin for p in page_statistics), default=0.0),
        indentation_levels=_indentation_levels(measurable, character_width),
        repeated_headers=repeated_headers,
        repeated_footers=repeated_footers,
        horizontal_rules=tuple(
            rule for page in document.pages for rule in page.horizontal_rules
        ),
        pages=page_statistics,
    )
    # Measured last because it needs the furniture test the rest of the
    # statistics provide, and nothing above it depends on the result.
    return replace(statistics, column_gutters=_column_gutters(document, statistics))
