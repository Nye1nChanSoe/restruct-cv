"""The physical representation of a document, read once.

Pass 1 records what is measurably on the page and nothing about what it means.
Every later pass reads this instead of re-opening the PDF, so a page is parsed,
rendered or OCR'd exactly once.

The native and OCR paths converge here. They differ in only one honest respect,
recorded as ``Span.granularity``: a native span knows its individual characters,
while OCR resolves no finer than a word. Nothing downstream branches on which
path produced a page -- it asks the granularity when it matters, which is only
during word reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

BBox = tuple[float, float, float, float]

# PyMuPDF span flag bits.
_FLAG_ITALIC = 1 << 1
_FLAG_SERIF = 1 << 2
_FLAG_MONOSPACED = 1 << 3
_FLAG_BOLD = 1 << 4


@dataclass(frozen=True)
class Token:
    """The smallest unit the source could report.

    Natively this is one character. From OCR it is one word, because Tesseract
    reports no finer geometry. ``Span.granularity`` says which, so no consumer
    has to guess.
    """

    text: str
    bbox: BBox
    origin: tuple[float, float] | None = None
    confidence: float | None = None
    # PyMuPDF inserts a synthetic glyph for things like a decomposed ligature.
    synthetic: bool = False

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass(frozen=True)
class Span:
    """A run of text sharing one font, size and style."""

    text: str
    bbox: BBox
    font: str
    size: float
    flags: int
    tokens: tuple[Token, ...] = ()
    granularity: Literal["character", "word"] = "character"
    color: int = 0
    origin: tuple[float, float] | None = None
    confidence: float | None = None

    @property
    def bold(self) -> bool:
        """Flagged bold, or named bold by a font that does not set the flag.

        Both tests are needed: some exporters encode weight only in the font
        name, and some only in the flag.
        """
        folded = self.font.casefold()
        return bool(self.flags & _FLAG_BOLD) or "bold" in folded or "black" in folded

    @property
    def italic(self) -> bool:
        return bool(self.flags & _FLAG_ITALIC) or "italic" in self.font.casefold()

    @property
    def serif(self) -> bool:
        return bool(self.flags & _FLAG_SERIF)

    @property
    def monospaced(self) -> bool:
        return bool(self.flags & _FLAG_MONOSPACED)

    @property
    def character_width(self) -> float:
        """Mean advance per character, measured rather than assumed.

        Pass 2 needs a spacing threshold relative to the current font and size;
        this is that measurement for this span.
        """
        if not self.text:
            return 0.0
        return (self.bbox[2] - self.bbox[0]) / len(self.text)


@dataclass(frozen=True)
class TextLine:
    """One physical line: the spans the source grouped onto a single baseline."""

    page: int
    bbox: BBox
    spans: tuple[Span, ...]
    # Writing direction as a unit vector; (1, 0) is ordinary left-to-right.
    direction: tuple[float, float] = (1.0, 0.0)
    used_ocr: bool = False

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def size(self) -> float:
        """The largest span size, which is what makes a line read as a heading."""
        return max((span.size for span in self.spans), default=0.0)

    @property
    def bold(self) -> bool:
        return any(span.bold for span in self.spans)

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_horizontal(self) -> bool:
        """Vertical text is outside v1 scope and must be detected, not parsed."""
        return abs(self.direction[0]) > abs(self.direction[1])


@dataclass(frozen=True)
class Rule:
    """A drawn line. Layout evidence, never on its own proof of a heading."""

    page: int
    bbox: BBox
    orientation: Literal["horizontal", "vertical"]
    thickness: float

    @property
    def length(self) -> float:
        if self.orientation == "horizontal":
            return self.bbox[2] - self.bbox[0]
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class ImageRegion:
    """An image placed on the page. A full-page one usually means a scan."""

    page: int
    bbox: BBox


@dataclass(frozen=True)
class LinkAnnotation:
    """A link rectangle and its target.

    The target is kept apart from the visible text because the two routinely
    disagree -- visible text abbreviates, the annotation carries the real URL.
    """

    page: int
    bbox: BBox
    uri: str


@dataclass(frozen=True)
class Page:
    """One page and everything found on it."""

    number: int
    width: float
    height: float
    rotation: int = 0
    used_ocr: bool = False
    lines: tuple[TextLine, ...] = ()
    rules: tuple[Rule, ...] = ()
    images: tuple[ImageRegion, ...] = ()
    links: tuple[LinkAnnotation, ...] = ()

    @property
    def horizontal_rules(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.orientation == "horizontal")

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True)
class Document:
    """Every page, read once, reused by every later pass."""

    pages: tuple[Page, ...] = ()
    # Untouched PyMuPDF and Tesseract output, retained only for debug dumps.
    raw_pages: tuple[dict, ...] = field(default=(), repr=False)
    ocr_pages: tuple[dict, ...] = field(default=(), repr=False)

    @property
    def lines(self) -> list[TextLine]:
        """Every line, in reading order across pages."""
        return [line for page in self.pages for line in page.lines]

    @property
    def used_ocr(self) -> bool:
        return any(page.used_ocr for page in self.pages)

    def page(self, number: int) -> Page:
        """One page by its 1-based number."""
        return self.pages[number - 1]

    def __iter__(self) -> Iterator[Page]:
        return iter(self.pages)
