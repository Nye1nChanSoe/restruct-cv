"""Adjustable settings for extraction, heading detection, and debug rendering."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSettings:
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class OcrSettings:
    enabled: bool = True
    language: str = "eng"
    dpi: int = 200
    native_text_min_characters: int = 20


@dataclass(frozen=True)
class HeadingSettings:
    similarity_threshold: float = 0.55
    winner_margin: float = 0.06
    maximum_words: int = 8
    maximum_characters: int = 80
    uppercase_ratio: float = 0.80
    font_size_multiplier: float = 1.12


@dataclass(frozen=True)
class DebugSettings:
    scale: float = 2.0
    heading_stroke_width: int = 5
    content_stroke_width: int = 3
    label_x_padding: int = 4
    label_y_offset: int = 14


SECTION_REFERENCES: dict[str, tuple[str, ...]] = {
    "summary": (
        "Professional Summary",
        "Career Profile",
        "Career Objective",
        "About Me",
    ),
    "experience": (
        "Work Experience",
        "Professional Experience",
        "Employment History",
        "Career History",
    ),
    "education": (
        "Education",
        "Academic Background",
        "Academic Qualifications",
    ),
    "skills": (
        "Skills",
        "Technical Skills",
        "Core Competencies",
        "Technologies and Tools",
    ),
    "projects": (
        "Projects",
        "Selected Projects",
        "Personal Projects",
        "Portfolio",
    ),
    "certifications": (
        "Certifications",
        "Certificates and Training",
        "Professional Certifications",
    ),
    "languages": (
        "Languages",
        "Language Proficiency",
    ),
    "volunteering": (
        "Volunteer Experience",
        "Community Involvement",
        "Community Service",
    ),
    "awards": (
        "Awards and Honors",
        "Achievements",
    ),
    "publications": (
        "Publications",
        "Research and Publications",
    ),
}

SECTION_COLORS: dict[str, str] = {
    "summary": "#2E7D32",
    "experience": "#1565C0",
    "education": "#7B1FA2",
    "skills": "#00897B",
    "projects": "#EF6C00",
    "certifications": "#C2185B",
    "languages": "#558B2F",
    "volunteering": "#00838F",
    "awards": "#F9A825",
    "publications": "#6D4C41",
}


@dataclass(frozen=True)
class ExtractorSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
    section_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(SECTION_REFERENCES)
    )
    section_colors: dict[str, str] = field(default_factory=lambda: dict(SECTION_COLORS))


SETTINGS = ExtractorSettings()
