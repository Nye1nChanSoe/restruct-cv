# Restruct accuracy scorecard

Scored against the hand-written labels in `tests/labels/`, which were
derived by reading each resume rather than from pipeline output. A span
counts as correct when it matches after normalization, or when one side
fully contains the other.

Resumes scored: 10

| Field | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `header.name` | 1.00 | 0.90 | 0.95 | 9 | 0 | 1 |
| `header.location` | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 |
| `header.nationality` | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| `header.job_titles` | 0.59 | 1.00 | 0.74 | 13 | 9 | 0 |
| `header.emails` | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 |
| `header.phones` | 1.00 | 0.89 | 0.94 | 8 | 0 | 1 |
| `header.urls` | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 |
| `section_routing` | 0.97 | 1.00 | 0.98 | 63 | 2 | 0 |
| `experience.job_titles` | 0.93 | 0.96 | 0.95 | 27 | 2 | 1 |
| `experience.companies` | 1.00 | 0.89 | 0.94 | 25 | 0 | 3 |
| `experience.dates` | 1.00 | 1.00 | 1.00 | 28 | 0 | 0 |
| `experience.locations` | 1.00 | 0.96 | 0.98 | 27 | 0 | 1 |
| `experience.entry_count` | 1.00 | 1.00 | 1.00 | 28 | 0 | 0 |
| `education.titles` | 0.92 | 1.00 | 0.96 | 12 | 1 | 0 |
| `education.institutions` | 0.92 | 1.00 | 0.96 | 12 | 1 | 0 |
| `education.dates` | 1.00 | 1.00 | 1.00 | 11 | 0 | 0 |
| `education.entry_count` | 0.92 | 1.00 | 0.96 | 12 | 1 | 0 |

**Macro F1 across scored fields: 0.963**

## Misses and spurious values

- `header.name`
  - missed: `6: alex morgan`
- `header.job_titles`
  - spurious: `1: mechanical & electrical maintenance`
  - spurious: `11: ux/ui`
  - spurious: `11: ui designer`
  - spurious: `12.ats: medical-surgical & post-operative care`
  - spurious: `2: inventory planning`
  - spurious: `2: data analysis`
  - spurious: `5.ocr: structural & construction works`
  - spurious: `6: alex morgan`
  - spurious: `6: backend & data systems`
- `header.phones`
  - missed: `5.ocr: +66 8x xxx xxxx`
- `section_routing`
  - spurious: `7.anomaly: languages`
  - spurious: `7.anomaly: others`
- `experience.job_titles`
  - missed: `11: junior product designer`
  - spurious: `12.ats: riverside general hospital`
  - spurious: `12.ats: riverside general hospital`
- `experience.companies`
  - missed: `11: northstar digital`
  - missed: `12.ats: riverside general hospital`
  - missed: `12.ats: riverside general hospital`
- `experience.locations`
  - missed: `9.ocr: jakarta, indonesia`
- `education.titles`
  - spurious: `7.anomaly: industrial mechanics / metal work (certificate level)`
- `education.institutions`
  - spurious: `7.anomaly: (fictional sample institution), bangkok`
- `education.entry_count`
  - spurious: `7.anomaly: entry-1`
