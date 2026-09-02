"""Build the metadata-free v1 resume output from inspectable parser results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


V1_SECTION_ORDER = (
    "header_profile",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "licenses",
    "tools_equipment",
    "languages",
    "volunteering",
    "awards",
    "publications",
    "references",
    "interests",
)


def _clean_text(value: Any) -> str:
    return str(value).replace("\u200b", "").replace("\ufeff", "").strip()


def _texts(values: list[dict[str, Any]]) -> list[str]:
    return [text for value in values if (text := _clean_text(value.get("text", "")))]


def _urls(values: list[dict[str, Any]]) -> list[dict[str, str]]:
    urls: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        text = _clean_text(value.get("text", ""))
        url = _clean_text(value.get("url", text))
        if not text and not url:
            continue
        key = (text, url)
        if key in seen:
            continue
        seen.add(key)
        urls.append({"text": text, "url": url})
    return urls


def _first_entity_text(
    entities: list[dict[str, Any]],
    entity_type: str,
) -> str | None:
    return next(
        (
            text
            for entity in entities
            if entity.get("type") == entity_type
            and (text := _clean_text(entity.get("text", "")))
        ),
        None,
    )


def _profile_value(header_profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if header_profile is None:
        return None
    entities = header_profile.get("entities", [])
    return {
        "name": _first_entity_text(entities, "name"),
        "job_titles": _texts(
            [entity for entity in entities if entity.get("type") == "job_title"]
        ),
        "location": _first_entity_text(entities, "location"),
        "nationality": _first_entity_text(entities, "nationality"),
        "emails": _texts(
            [entity for entity in entities if entity.get("type") == "email"]
        ),
        "phones": _texts(
            [entity for entity in entities if entity.get("type") == "phone"]
        ),
        "urls": _urls(
            [entity for entity in entities if entity.get("type") == "url"]
        ),
    }


def _summary_value(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "content": [
            {
                "type": str(item["type"]),
                "text": _clean_text(item.get("text", "")),
            }
            for item in summary.get("content", [])
            if _clean_text(item.get("text", ""))
        ]
    }


def _experience_value(experience: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if experience is None:
        return None
    return [
        {
            "job_titles": _texts(entry.get("jobTitles", [])),
            "companies": _texts(entry.get("companies", [])),
            "dates": _texts(entry.get("dates", [])),
            "locations": _texts(entry.get("locations", [])),
            "urls": _urls(entry.get("urls", [])),
            "paragraphs": _texts(entry.get("paragraphs", [])),
            "bullets": _texts(entry.get("bullets", [])),
        }
        for entry in experience.get("entries", [])
    ]


def _education_value(education: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if education is None:
        return None
    return [
        {
            "titles": _texts(entry.get("titles", [])),
            "institutions": _texts(entry.get("institutions", [])),
            "dates": _texts(entry.get("dates", [])),
            "locations": _texts(entry.get("locations", [])),
            "gpa": _texts(entry.get("gpa", [])),
            "skills": _texts(entry.get("skills", [])),
            "urls": _urls(entry.get("urls", [])),
            "paragraphs": _texts(entry.get("paragraphs", [])),
            "bullets": _texts(entry.get("bullets", [])),
        }
        for entry in education.get("entries", [])
    ]


def _skills_value(skills: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if skills is None:
        return None
    return [
        {
            "subheading": (
                _clean_text(group["subheading"].get("text", ""))
                if group.get("subheading") is not None
                else None
            ),
            "paragraphs": _texts(group.get("paragraphs", [])),
            "bullets": _texts(group.get("bullets", [])),
            "urls": _urls(group.get("urls", [])),
        }
        for group in skills.get("groups", [])
    ]


def _grouped_section_value(
    section: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if section is None:
        return None
    return [
        {
            "subheadings": _texts(entry.get("subheadingLines", [])),
            "dates": _texts(entry.get("dates", [])),
            "urls": _urls(entry.get("urls", [])),
            "paragraphs": _texts(entry.get("paragraphs", [])),
            "bullets": _texts(entry.get("bullets", [])),
        }
        for entry in section.get("entries", [])
    ]


def build_v1_resume(
    *,
    header_profile: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    experience: dict[str, Any] | None,
    education: dict[str, Any] | None,
    skills: dict[str, Any] | None,
    projects: dict[str, Any] | None,
    supplementary_sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "header_profile": _profile_value(header_profile),
        "summary": _summary_value(summary),
        "experience": _experience_value(experience),
        "education": _education_value(education),
        "skills": _skills_value(skills),
        "projects": _grouped_section_value(projects),
    }
    for section_type in V1_SECTION_ORDER[6:]:
        values[section_type] = _grouped_section_value(
            supplementary_sections.get(section_type)
        )
    return {section_type: values.get(section_type) for section_type in V1_SECTION_ORDER}


def write_v1_resume(output_directory: Path, resume: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "resume.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
