# Restruct accuracy scorecard

Scored against the hand-written labels in `tests/labels/`, which were
derived by reading each resume rather than from pipeline output. A span
counts as correct when it matches after normalization, or when one side
fully contains the other.

Resumes scored: 9

| Field | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `header.name` | 1.00 | 0.89 | 0.94 | 8 | 0 | 1 |
| `header.location` | 1.00 | 1.00 | 1.00 | 9 | 0 | 0 |
| `header.nationality` | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| `header.job_titles` | 0.60 | 1.00 | 0.75 | 12 | 8 | 0 |
| `header.emails` | 1.00 | 1.00 | 1.00 | 9 | 0 | 0 |
| `header.phones` | 1.00 | 0.88 | 0.93 | 7 | 0 | 1 |
| `header.urls` | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 |
| `section_routing` | 0.97 | 1.00 | 0.98 | 57 | 2 | 0 |
| `experience.job_titles` | 1.00 | 0.96 | 0.98 | 24 | 0 | 1 |
| `experience.companies` | 1.00 | 0.96 | 0.98 | 24 | 0 | 1 |
| `experience.dates` | 1.00 | 1.00 | 1.00 | 25 | 0 | 0 |
| `experience.locations` | 1.00 | 0.96 | 0.98 | 24 | 0 | 1 |
| `experience.entry_count` | 1.00 | 1.00 | 1.00 | 25 | 0 | 0 |
| `education.titles` | 0.92 | 1.00 | 0.96 | 11 | 1 | 0 |
| `education.institutions` | 0.92 | 1.00 | 0.96 | 11 | 1 | 0 |
| `education.dates` | 1.00 | 1.00 | 1.00 | 10 | 0 | 0 |
| `education.entry_count` | 0.92 | 1.00 | 0.96 | 11 | 1 | 0 |

**Macro F1 across scored fields: 0.966**

## Misses and spurious values

- `header.name`
  - missed: `6: alex morgan`
- `header.job_titles`
  - spurious: `1: mechanical & electrical maintenance`
  - spurious: `11: ux/ui`
  - spurious: `11: ui designer`
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
- `experience.companies`
  - missed: `11: northstar digital`
- `experience.locations`
  - missed: `9.ocr: jakarta, indonesia`
- `education.titles`
  - spurious: `7.anomaly: industrial mechanics / metal work (certificate level)`
- `education.institutions`
  - spurious: `7.anomaly: (fictional sample institution), bangkok`
- `education.entry_count`
  - spurious: `7.anomaly: entry-1`
