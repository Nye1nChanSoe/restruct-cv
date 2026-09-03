"""Adjustable settings for extraction, semantic detection, and debug rendering."""

from dataclasses import dataclass, field

from extractor_v1.configs.embedding_references import (
    JOB_TITLE_NEGATIVE_REFERENCES,
    JOB_TITLE_REFERENCES,
    PROFILE_ATTRIBUTE_REFERENCES,
    SECTION_COLORS,
    SECTION_REFERENCES,
)


@dataclass(frozen=True)
class ModelSettings:
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    local_directory: str = "models/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class NerSettings:
    distilbert_name: str = "dslim/distilbert-NER"
    distilbert_revision: str = "dfa2838a127384aabb82ed7719e16dab84c42a2a"
    distilbert_local_directory: str = "models/distilbert-NER"
    minimum_confidence: float = 0.50
    labels: tuple[str, ...] = (
        "person name",
        "location",
        "nationality",
    )


@dataclass(frozen=True)
class OcrSettings:
    enabled: bool = True
    language: str = "eng"
    dpi: int = 300
    engine_mode: int = 1
    page_segmentation_mode: int = 3
    tesseract_command: str = "tesseract"
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
class SectionRouterSettings:
    subheading_font_size_multiplier: float = 1.08
    maximum_subheading_words: int = 12
    maximum_subheading_characters: int = 100
    paragraph_gap_multiplier: float = 1.25


@dataclass(frozen=True)
class UrlSettings:
    annotation_bbox_tolerance: float = 2.0


@dataclass(frozen=True)
class HeaderProfileSettings:
    boundary_similarity_threshold: float = 0.72
    boundary_winner_margin: float = 0.08
    maximum_lines_without_boundary: int = 16


@dataclass(frozen=True)
class JobTitleSettings:
    similarity_threshold: float = 0.48
    winner_margin: float = 0.06
    maximum_words: int = 10
    maximum_segments_per_line: int = 3


@dataclass(frozen=True)
class ProfileAttributeSettings:
    similarity_threshold: float = 0.62
    winner_margin: float = 0.08


@dataclass(frozen=True)
class DebugSettings:
    raw_extraction_enabled: bool = True
    raw_extraction_directory: str = "debug"
    ocr_extraction_enabled: bool = True
    ocr_extraction_directory: str = "debug/ocr"
    scale: float = 2.0
    heading_stroke_width: int = 5
    content_stroke_width: int = 3
    label_x_padding: int = 4
    label_y_offset: int = 14
    header_region_color: str = "#37474F"
    header_region_stroke_width: int = 6
    header_entity_stroke_width: int = 4
    header_entity_colors: tuple[tuple[str, str], ...] = (
        ("name", "#D32F2F"),
        ("job_title", "#7B1FA2"),
        ("location", "#EF6C00"),
        ("nationality", "#F9A825"),
        ("current_residence", "#FB8C00"),
        ("date_of_birth", "#8E24AA"),
        ("age", "#AB47BC"),
        ("gender", "#00838F"),
        ("marital_status", "#5E35B1"),
        ("visa_status", "#3949AB"),
        ("email", "#00897B"),
        ("phone", "#1565C0"),
        ("url", "#6D4C41"),
        ("other", "#D3D3D3"),
    )


@dataclass(frozen=True)
class PathSettings:
    input_directory: str = "resumes-synthetic"
    truths_input_directory: str = "resumes-truths"
    results_directory: str = "results"
    truths_results_directory: str = "results/0-truths"


@dataclass(frozen=True)
class ExtractorSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    ner: NerSettings = field(default_factory=NerSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    section_router: SectionRouterSettings = field(default_factory=SectionRouterSettings)
    url: UrlSettings = field(default_factory=UrlSettings)
    header_profile: HeaderProfileSettings = field(default_factory=HeaderProfileSettings)
    job_title: JobTitleSettings = field(default_factory=JobTitleSettings)
    profile_attribute: ProfileAttributeSettings = field(
        default_factory=ProfileAttributeSettings
    )
    debug: DebugSettings = field(default_factory=DebugSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    section_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(SECTION_REFERENCES)
    )
    section_colors: dict[str, str] = field(default_factory=lambda: dict(SECTION_COLORS))
    job_title_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(JOB_TITLE_REFERENCES)
    )
    job_title_negative_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(JOB_TITLE_NEGATIVE_REFERENCES)
    )
    profile_attribute_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(PROFILE_ATTRIBUTE_REFERENCES)
    )


SETTINGS = ExtractorSettings()
