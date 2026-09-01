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
    raw_extraction_enabled: bool = True
    raw_extraction_directory: str = "debug"
    scale: float = 2.0
    heading_stroke_width: int = 5
    content_stroke_width: int = 3
    label_x_padding: int = 4
    label_y_offset: int = 14


@dataclass(frozen=True)
class PathSettings:
    input_directory: str = "resumes-synthetic"
    truths_input_directory: str = "truths"
    results_directory: str = "results"
    truths_results_directory: str = "results/truths"


SECTION_REFERENCES: dict[str, tuple[str, ...]] = {
    "summary": (
        "Professional Summary",
        "Career Profile",
        "Career Objective",
        "About Me",
        "Summary",
        "Profile",
        "Executive Summary",
        "Personal Statement",
        "Objective",
        "Professional Profile",
        "Career Summary",
        "Overview",
        "Highlights",
        "Key Qualifications",
    ),
    "experience": (
        "Work Experience",
        "Professional Experience",
        "Employment History",
        "Career History",
        "Experience",
        "Relevant Experience",
        "Work History",
        "Job History",
        "Employment",
        "Positions Held",
        "Field Experience",
        "Industry Experience",
        "Design Experience",
        "Marketing Experience",
        "Freelance Experience",
        "Client Experience",
        "Internship Experience",
        "Practical Experience",
    ),
    "education": (
        "Education",
        "Academic Background",
        "Academic Qualifications",
        "Education and Training",
        "Educational Background",
        "Academic History",
        "Schooling",
        "Degrees",
        "Coursework",
        "Relevant Coursework",
    ),
    "skills": (
        "Skills",
        "Technical Skills",
        "Core Competencies",
        "Technologies and Tools",
        "Key Skills",
        "Skill Set",
        "Areas of Expertise",
        "Competencies",
        "Software Skills",
        "Design Skills",
        "Design Tools",
        "Programming Languages",
        "Languages and Frameworks",
        "Tech Stack",
        "Tools and Technologies",
        "Marketing Skills",
        "Digital Skills",
        "Hard Skills",
        "Soft Skills",
        "Proficiencies",
        "Specializations",
        "UX Skills",
        "UI Skills",
        "Skills Summary",
    ),
    "projects": (
        "Projects",
        "Selected Projects",
        "Personal Projects",
        "Portfolio",
        "Case Studies",
        "Design Portfolio",
        "Featured Projects",
        "Key Projects",
        "Project Highlights",
        "Campaigns",
        "Selected Campaigns",
        "Notable Work",
        "Side Projects",
        "Open Source Contributions",
        "GitHub Projects",
        "Academic Projects",
    ),
    "certifications": (
        "Certifications",
        "Certificates and Training",
        "Professional Certifications",
        "Certificates",
        "Licenses and Certifications",
        "Training and Certifications",
        "Professional Development",
        "Credentials",
        "Accreditations",
    ),
    "licenses": (
        "Licenses",
        "Driver's License",
        "Commercial License",
        "Trade License",
        "Permits and Licenses",
        "Professional Licenses",
        "CDL",
        "OSHA Certification",
        "Safety Certifications",
        "Forklift Certification",
    ),
    "tools_equipment": (
        "Tools and Equipment",
        "Equipment Operated",
        "Machinery",
        "Equipment Proficiency",
        "Tools",
        "Equipment Experience",
        "Machine Operation",
    ),
    "languages": (
        "Languages",
        "Language Proficiency",
        "Language Skills",
        "Spoken Languages",
    ),
    "volunteering": (
        "Volunteer Experience",
        "Community Involvement",
        "Community Service",
        "Volunteering",
        "Volunteer Work",
        "Civic Engagement",
    ),
    "awards": (
        "Awards and Honors",
        "Achievements",
        "Awards",
        "Honors",
        "Recognition",
        "Accolades",
        "Accomplishments",
    ),
    "publications": (
        "Publications",
        "Research and Publications",
        "Papers",
        "Articles Published",
        "Conference Talks",
        "Speaking Engagements",
        "Presentations",
    ),
    "references": (
        "References",
        "Professional References",
        "References Available Upon Request",
    ),
    "interests": (
        "Interests",
        "Hobbies",
        "Personal Interests",
        "Activities",
        "Extracurricular Activities",
    ),
}

SECTION_COLORS: dict[str, str] = {
    "summary":         "#2E7D32",  # green — intro/overview, calm anchor color
    "experience":      "#1565C0",  # blue — core professional history, primary section
    "education":       "#7B1FA2",  # purple — academic, distinct from work color
    "skills":          "#00897B",  # teal — competencies, sits between blue/green
    "projects":        "#EF6C00",  # orange — hands-on work, high visual energy
    "certifications":  "#C2185B",  # pink/magenta — formal credentials
    "licenses":        "#AD1457",  # deep magenta — related to certs but distinct (legal/regulatory)
    "tools_equipment":  "#00695C",  # dark teal — adjacent to skills, trades-specific
    "languages":       "#558B2F",  # olive green — near summary but clearly separate
    "volunteering":    "#00838F",  # cyan — community/civic, distinct from experience blue
    "awards":          "#F9A825",  # amber/gold — recognition, intuitive gold association
    "publications":    "#6D4C41",  # brown — academic/print association
    "references":      "#455A64",  # blue-grey — neutral, low-emphasis (usually sparse content)
    "interests":       "#8D6E63",  # warm taupe — personal/light-weight section
}


@dataclass(frozen=True)
class ExtractorSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    section_references: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(SECTION_REFERENCES)
    )
    section_colors: dict[str, str] = field(default_factory=lambda: dict(SECTION_COLORS))


SETTINGS = ExtractorSettings()
