"""Draw a resume back out of ``resume.json``, as a page a person can read.

The overlays answer a geometric question: did this box land on the right words.
They cannot answer the other one, because they draw on top of the document and
the document keeps making sense regardless of what was understood. A bullet
filed under education, a date read as a job title, a section that swallowed the
one after it -- every box is in the right place, and the overlay looks correct.

So this throws the page away and draws only what was understood. Reading it is
proof-reading: anything that comes out looking wrong *is* wrong, because there
is nothing else left in the picture to be wrong.

Two rules follow from that:

- It renders from the written ``resume.json`` and nothing else -- not from the
  parser's own types, which would let it show something the file does not
  contain. That makes it a consumer of the published contract, so a field the
  contract cannot express is a field this cannot draw.
- It is **not** a facsimile. Imitating the original layout would hide exactly
  the errors it exists to reveal, and for a DOCX it would mean inventing the
  geometry `ingestion/docx.py` refuses to invent.

Empty and absent values are skipped entirely: a page of "none" rows is a page
nobody proof-reads. What must never be skipped is content with nowhere to go --
a schema field this module has not been taught to draw would otherwise vanish
silently, so it is collected and drawn under UNPLACED in red.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pymupdf

# A4, in points. The reconstruction has no page size of its own to inherit --
# it is drawn from the JSON, which records none -- so it picks one and keeps it
# the same for every document, which also makes two of them comparable.
PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN = 54.0

_TEXT_COLOR = (0.07, 0.07, 0.08)
_MUTED_COLOR = (0.38, 0.39, 0.42)
_RULE_COLOR = (0.78, 0.79, 0.82)
_UNPLACED_COLOR = (0.72, 0.11, 0.11)

# Weight is drawn, not chosen: the one font that covers the document's
# characters may have no bold face, and losing a Thai glyph to gain a heavier
# heading is a bad trade in a tool for reading. Stroking the glyph outline
# alongside the fill thickens any face by a controlled amount.
_BOLD_STROKE = 0.045


@dataclass(frozen=True)
class Style:
    """How one kind of line is drawn, and what space it keeps around itself."""

    size: float
    leading: float
    space_before: float = 0.0
    space_after: float = 0.0
    bold: bool = False
    color: tuple[float, float, float] = _TEXT_COLOR
    indent: float = 0.0
    marker: str = ""
    uppercase: bool = False
    rule: bool = False


STYLES: dict[str, Style] = {
    "name": Style(size=20, leading=24, bold=True, space_after=2),
    "tagline": Style(size=11, leading=14, color=_MUTED_COLOR, space_after=1),
    "contact": Style(size=9, leading=12, color=_MUTED_COLOR),
    "heading": Style(
        size=11.5,
        leading=15,
        bold=True,
        space_before=16,
        space_after=7,
        uppercase=True,
        rule=True,
    ),
    "subheading": Style(size=11, leading=14, bold=True, space_before=9, space_after=1),
    "meta": Style(size=9.5, leading=12.5, color=_MUTED_COLOR, space_after=3),
    "paragraph": Style(size=10, leading=13.5, space_after=4),
    "bullet": Style(size=10, leading=13.5, indent=14, marker="\u2022", space_after=2),
    "label": Style(size=10, leading=13.5, space_after=2),
    "unplaced_heading": Style(
        size=11.5,
        leading=15,
        bold=True,
        space_before=16,
        space_after=7,
        uppercase=True,
        color=_UNPLACED_COLOR,
        rule=True,
    ),
    "unplaced": Style(size=10, leading=13.5, color=_UNPLACED_COLOR, space_after=2),
}


@dataclass
class Block:
    """One styled line of text, before it knows which page it lands on."""

    style: str
    text: str


# What the builder below reads. A key outside this set is content the renderer
# has never been taught to draw, which is the case UNPLACED exists for.
_KNOWN_KEYS = frozenset(
    {
        "schema_version",
        # header
        "name",
        "job_titles",
        "location",
        "date_of_birth",
        "age",
        "gender",
        "marital_status",
        "visa_status",
        "nationality",
        "current_residence",
        "current_income",
        "current_package",
        "emails",
        "phones",
        "urls",
        # entries
        "content",
        "type",
        "text",
        "url",
        "value",
        "companies",
        "dates",
        "locations",
        "paragraphs",
        "bullets",
        "titles",
        "institutions",
        "gpa",
        "skills",
        "subheading",
        "subheadings",
        "attributes",
        "heading",
        "entries",
    }
)

# The sixteen destinations, plus the version. Everything under them is drawn
# by one of the builders below; a key that is neither is content this renderer
# has never been taught to place.
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "header_profile", "others"}
)

_SECTION_TITLES = {
    "summary": "Summary",
    "experience": "Experience",
    "education": "Education",
    "skills": "Skills",
    "projects": "Projects",
    "certifications": "Certifications",
    "licenses": "Licenses",
    "tools_equipment": "Tools & Equipment",
    "languages": "Languages",
    "volunteering": "Volunteering",
    "awards": "Awards",
    "publications": "Publications",
    "references": "References",
    "interests": "Interests",
}

_DRAWN_KEYS = _KNOWN_KEYS | _TOP_LEVEL_KEYS | frozenset(_SECTION_TITLES)

_PROFILE_LABELS = (
    ("date_of_birth", "Date of birth"),
    ("age", "Age"),
    ("gender", "Gender"),
    ("marital_status", "Marital status"),
    ("visa_status", "Visa status"),
    ("nationality", "Nationality"),
    ("current_residence", "Current residence"),
    ("current_income", "Current income"),
    ("current_package", "Current package"),
)

_SEPARATOR = "  \u00b7  "


# -- the font ---------------------------------------------------------------


def _font_candidates() -> tuple[str, ...]:
    """System fonts to try, widest coverage first.

    A resume is not Latin-1: the corpus already carries en dashes, curly quotes
    and Thai. PDF's base-14 fonts draw every one of those as a middle dot,
    which in a proof-reading tool is worse than useless -- the reader would
    chase an extraction bug that is really a missing glyph.
    """
    if sys.platform == "darwin":
        return (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        )
    if sys.platform == "win32":
        root = os.environ.get("WINDIR", "C:\\Windows")
        return (
            os.path.join(root, "Fonts", "arialuni.ttf"),
            os.path.join(root, "Fonts", "segoeui.ttf"),
            os.path.join(root, "Fonts", "arial.ttf"),
        )
    return (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    )


@dataclass(frozen=True)
class ChosenFont:
    """The face to draw with, and the characters it cannot draw."""

    font: pymupdf.Font
    path: str | None
    missing: tuple[str, ...] = ()

    # What the page calls the embedded font. A PDF resource name may not carry
    # a space, and several of the faces worth using ("Arial Unicode") do.
    resource_name: str = "reconstruction"

    @property
    def name(self) -> str:
        return Path(self.path).stem if self.path else "Helvetica (built-in)"


def choose_font(characters: set[str]) -> ChosenFont:
    """The first installed font that can draw this document, or the closest.

    Chosen against the text about to be drawn rather than by name, because
    which font covers a document is a property of the document. When nothing
    covers it, the least-bad face is used and the missing characters are
    reported so the page can say so out loud -- an unreadable glyph must never
    be mistaken for an extraction error.
    """
    wanted = {character for character in characters if not character.isspace()}
    best: ChosenFont | None = None
    for path in _font_candidates():
        if not os.path.exists(path):
            continue
        try:
            font = pymupdf.Font(fontfile=path)
        except Exception:  # an unreadable or unsupported font file
            continue
        missing = tuple(
            sorted(character for character in wanted if not font.has_glyph(ord(character)))
        )
        candidate = ChosenFont(font=font, path=path, missing=missing)
        if not missing:
            return candidate
        if best is None or len(missing) < len(best.missing):
            best = candidate
    if best is not None:
        return best
    font = pymupdf.Font("helv")
    missing = tuple(
        sorted(character for character in wanted if not font.has_glyph(ord(character)))
    )
    return ChosenFont(font=font, path=None, missing=missing)


# -- turning the file into blocks -------------------------------------------


def _joined(values: Any, separator: str = _SEPARATOR) -> str:
    return separator.join(str(value) for value in values if str(value).strip())


def _url_texts(urls: Any) -> list[str]:
    """A URL as one string, without repeating the text when it is the URL."""
    texts: list[str] = []
    for entry in urls or []:
        text = str(entry.get("text", "")).strip()
        url = str(entry.get("url", "")).strip()
        texts.append(text if text == url or not url else (f"{text} ({url})" if text else url))
    return texts


def _entry_body(entry: dict[str, Any]) -> Iterator[Block]:
    """The paragraphs, bullets and links every entry shape shares."""
    for paragraph in entry.get("paragraphs") or []:
        yield Block("paragraph", str(paragraph))
    for bullet in entry.get("bullets") or []:
        yield Block("bullet", str(bullet))
    links = _url_texts(entry.get("urls"))
    if links:
        yield Block("meta", _joined(links))


def _header_blocks(profile: dict[str, Any]) -> Iterator[Block]:
    name = str(profile.get("name") or "").strip()
    if name:
        yield Block("name", name)
    titles = _joined(profile.get("job_titles") or [])
    if titles:
        yield Block("tagline", titles)
    contact = [
        value
        for value in (
            str(profile.get("location") or "").strip(),
            _joined(profile.get("emails") or []),
            _joined(profile.get("phones") or []),
            _joined(_url_texts(profile.get("urls"))),
        )
        if value
    ]
    if contact:
        yield Block("contact", _joined(contact))
    for key, label in _PROFILE_LABELS:
        value = str(profile.get(key) or "").strip()
        if value:
            yield Block("contact", f"{label}: {value}")


def _summary_blocks(summary: dict[str, Any]) -> Iterator[Block]:
    for item in summary.get("content") or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        yield Block("bullet" if item.get("type") == "bullet" else "paragraph", text)


def _experience_blocks(entries: list[dict[str, Any]]) -> Iterator[Block]:
    for entry in entries:
        titles = _joined(entry.get("job_titles") or [], " / ")
        if titles:
            yield Block("subheading", titles)
        meta = _joined(
            [
                _joined(entry.get("companies") or []),
                _joined(entry.get("locations") or []),
                _joined(entry.get("dates") or []),
            ]
        )
        if meta:
            yield Block("meta", meta)
        yield from _entry_body(entry)


def _education_blocks(entries: list[dict[str, Any]]) -> Iterator[Block]:
    for entry in entries:
        titles = _joined(entry.get("titles") or [], " / ")
        if titles:
            yield Block("subheading", titles)
        meta = _joined(
            [
                _joined(entry.get("institutions") or []),
                _joined(entry.get("locations") or []),
                _joined(entry.get("dates") or []),
                _joined(entry.get("gpa") or []),
            ]
        )
        if meta:
            yield Block("meta", meta)
        skills = _joined(entry.get("skills") or [], ", ")
        if skills:
            yield Block("paragraph", skills)
        yield from _entry_body(entry)


def _skills_blocks(groups: list[dict[str, Any]]) -> Iterator[Block]:
    for group in groups:
        subheading = str(group.get("subheading") or "").strip()
        if subheading:
            yield Block("subheading", subheading)
        yield from _entry_body(group)


def _grouped_blocks(entries: list[dict[str, Any]]) -> Iterator[Block]:
    for entry in entries:
        subheadings = _joined(entry.get("subheadings") or [], " / ")
        if subheadings:
            yield Block("subheading", subheadings)
        dates = _joined(entry.get("dates") or [])
        if dates:
            yield Block("meta", dates)
        for attribute in entry.get("attributes") or []:
            value = str(attribute.get("value") or "").strip()
            if value:
                yield Block("label", f"{attribute.get('type', 'attribute')}: {value}")
        yield from _entry_body(entry)


def _others_blocks(sections: list[dict[str, Any]]) -> Iterator[Block]:
    for section in sections:
        # The heading as the document wrote it, which is the whole point of
        # `others`: it is where a section goes when nothing named it.
        heading = str(section.get("heading") or "").strip() or "Other"
        yield Block("heading", heading)
        yield from _grouped_blocks(section.get("entries") or [])


def _unplaced(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Content under a key this renderer does not know how to draw.

    Without this, adding a field to the schema and forgetting to teach the
    renderer would leave the field invisible -- and an invisible field in a
    proof-reading tool reads as an extraction failure.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key in _DRAWN_KEYS:
                yield from _unplaced(item, here)
            elif item not in (None, [], {}, ""):
                yield here, item
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _unplaced(item, f"{path}[{index}]")


def resume_blocks(resume: dict[str, Any]) -> list[Block]:
    """Every line to be drawn, in order, skipping everything absent or empty."""
    blocks: list[Block] = []
    profile = resume.get("header_profile")
    if isinstance(profile, dict):
        blocks.extend(_header_blocks(profile))

    summary = resume.get("summary")
    if isinstance(summary, dict) and (summary.get("content") or []):
        blocks.append(Block("heading", _SECTION_TITLES["summary"]))
        blocks.extend(_summary_blocks(summary))

    builders = {
        "experience": _experience_blocks,
        "education": _education_blocks,
        "skills": _skills_blocks,
    }
    for section, builder in builders.items():
        entries = resume.get(section)
        if entries:
            section_blocks = list(builder(entries))
            if section_blocks:
                blocks.append(Block("heading", _SECTION_TITLES[section]))
                blocks.extend(section_blocks)

    for section, title in _SECTION_TITLES.items():
        if section in builders or section == "summary":
            continue
        entries = resume.get(section)
        if entries:
            section_blocks = list(_grouped_blocks(entries))
            if section_blocks:
                blocks.append(Block("heading", title))
                blocks.extend(section_blocks)

    others = resume.get("others")
    if others:
        blocks.extend(_others_blocks(others))

    unplaced = list(_unplaced(resume))
    if unplaced:
        blocks.append(Block("unplaced_heading", "Unplaced"))
        blocks.extend(Block("unplaced", f"{path}: {item}") for path, item in unplaced)
    return blocks


# -- drawing ----------------------------------------------------------------


@dataclass
class _Cursor:
    """Where the next line goes, and which page it goes on.

    The page is fetched by number rather than held, because adding a page
    invalidates every Page object already taken from the document -- a held
    reference goes on looking usable and fails at the next draw.
    """

    document: pymupdf.Document
    page_count: int = 0
    y: float = MARGIN

    @property
    def page(self) -> Any:
        return self.document[self.page_count - 1]

    def new_page(self) -> None:
        self.document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.page_count += 1
        self.y = MARGIN

    def room_for(self, height: float) -> None:
        if self.page_count == 0 or self.y + height > PAGE_HEIGHT - MARGIN:
            self.new_page()


def _wrap(text: str, font: pymupdf.Font, size: float, width: float) -> list[str]:
    """Greedy word wrap, measured in the font the line is drawn with."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and font.text_length(candidate, size) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines or [""]


def _draw_block(cursor: _Cursor, block: Block, chosen: ChosenFont) -> None:
    style = STYLES[block.style]
    text = block.text.upper() if style.uppercase else block.text
    left = MARGIN + style.indent
    width = PAGE_WIDTH - MARGIN - left
    lines = _wrap(text, chosen.font, style.size, width)

    cursor.y += style.space_before
    # A heading keeps its first line with the block under it; a heading alone
    # at the foot of a page is a reading error the tool would have introduced.
    keep_with_next = style.leading * 2 if style.rule or style.bold else 0.0
    cursor.room_for(style.leading + keep_with_next)

    for index, line in enumerate(lines):
        cursor.room_for(style.leading)
        baseline = cursor.y + style.size
        if index == 0 and style.marker:
            cursor.page.insert_text(
                (MARGIN, baseline),
                style.marker,
                fontsize=style.size,
                fontname=chosen.resource_name,
                fontfile=chosen.path,
                color=style.color,
            )
        cursor.page.insert_text(
            (left, baseline),
            line,
            fontsize=style.size,
            fontname=chosen.resource_name,
            fontfile=chosen.path,
            color=style.color,
            render_mode=2 if style.bold else 0,
            border_width=_BOLD_STROKE if style.bold else 0,
            stroke_opacity=1,
        )
        cursor.y += style.leading

    if style.rule:
        rule_y = cursor.y - style.leading + style.size + 3
        cursor.page.draw_line(
            (MARGIN, rule_y),
            (PAGE_WIDTH - MARGIN, rule_y),
            color=_RULE_COLOR,
            width=0.6,
        )
    cursor.y += style.space_after


def _draw_footer(page: Any, number: int, total: int, chosen: ChosenFont, note: str) -> None:
    """Say what the page is, on the page.

    Printed and passed around, a reconstruction is indistinguishable from a
    resume. It should never be mistaken for the document it was drawn from.
    """
    label = f"restruct reconstruction \u2014 page {number} of {total}"
    if note:
        label = f"{label}   {note}"
    page.insert_text(
        (MARGIN, PAGE_HEIGHT - MARGIN + 22),
        label,
        fontsize=7.5,
        fontname=chosen.resource_name,
        fontfile=chosen.path,
        color=_MUTED_COLOR,
    )


def _characters(blocks: list[Block]) -> set[str]:
    return {character for block in blocks for character in block.text}


def render_resume(
    resume: dict[str, Any],
    output_directory: Path,
    *,
    prefix: str = "",
) -> list[Path]:
    """Draw one resume into ``reconstruction.pdf`` and one PNG per page.

    Both, because they answer different needs: the PDF is what a person reads
    and annotates, and the PNGs sit beside ``debug/page-N.png`` so the
    reconstruction and the overlay of the same document can be read side by
    side.

    ``prefix`` is what lets the two file names be written flat into a directory
    that holds other things: the names are fixed, so a caller that is not
    writing into a directory of its own has to qualify them itself.
    """
    blocks = resume_blocks(resume)
    chosen = choose_font(_characters(blocks))
    document = pymupdf.open()
    cursor = _Cursor(document=document)
    cursor.new_page()
    for block in blocks:
        _draw_block(cursor, block, chosen)

    note = (
        f"some characters are missing from {chosen.name}: {''.join(chosen.missing)}"
        if chosen.missing
        else ""
    )
    for number in range(1, cursor.page_count + 1):
        _draw_footer(document[number - 1], number, cursor.page_count, chosen, note)

    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdf_path = output_directory / f"{prefix}reconstruction.pdf"
    # A font chosen for coverage is a large file -- Arial Unicode is 23MB --
    # and embedding it whole put all of it in every drawing, which made a
    # two-page resume too big to open on a phone or send anywhere. Subsetting
    # keeps only the glyphs actually drawn: 23MB becomes about 60KB.
    document.subset_fonts(verbose=False)
    document.save(str(pdf_path), garbage=4, deflate=True, clean=True)
    written.append(pdf_path)
    for number in range(1, cursor.page_count + 1):
        image_path = output_directory / f"{prefix}page-{number}.png"
        document[number - 1].get_pixmap(dpi=144).save(str(image_path))
        written.append(image_path)
    document.close()
    return written


def render_resume_file(
    resume_path: Path,
    output_directory: Path,
    *,
    prefix: str = "",
) -> list[Path]:
    """Draw the resume written at ``resume_path``.

    Reading the file back, rather than taking the dictionary from the pipeline
    that just built it, is deliberate: it is the only way this stays a test of
    what was actually published.
    """
    import json

    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    return render_resume(resume, output_directory, prefix=prefix)
