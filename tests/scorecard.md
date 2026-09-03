# Restruct accuracy scorecard

Scored against the hand-written labels in `tests/labels/`, which were
derived by reading each resume rather than from pipeline output. A span
counts as correct when it matches after normalization, or when one side
fully contains the other.

Resumes scored: 6

| Field | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `header.name` | 1.00 | 0.83 | 0.91 | 5 | 0 | 1 |
| `header.location` | 1.00 | 1.00 | 1.00 | 6 | 0 | 0 |
| `header.nationality` | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| `header.job_titles` | 0.60 | 1.00 | 0.75 | 9 | 6 | 0 |
| `header.emails` | 1.00 | 1.00 | 1.00 | 6 | 0 | 0 |
| `header.phones` | 1.00 | 0.83 | 0.91 | 5 | 0 | 1 |
| `header.urls` | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 |
| `section_routing` | 0.93 | 0.95 | 0.94 | 38 | 3 | 2 |
| `experience.job_titles` | 0.95 | 1.00 | 0.97 | 18 | 1 | 0 |
| `experience.companies` | 1.00 | 0.94 | 0.97 | 17 | 0 | 1 |
| `experience.dates` | 1.00 | 1.00 | 1.00 | 18 | 0 | 0 |
| `experience.locations` | 1.00 | 0.94 | 0.97 | 17 | 0 | 1 |
| `experience.entry_count` | 1.00 | 1.00 | 1.00 | 18 | 0 | 0 |
| `education.titles` | 0.89 | 1.00 | 0.94 | 8 | 1 | 0 |
| `education.institutions` | 0.89 | 1.00 | 0.94 | 8 | 1 | 0 |
| `education.dates` | 1.00 | 1.00 | 1.00 | 7 | 0 | 0 |
| `education.entry_count` | 0.89 | 1.00 | 0.94 | 8 | 1 | 0 |

**Macro F1 across scored fields: 0.956**

## Misses and spurious values

- `header.name`
  - missed: `6: alex morgan`
- `header.job_titles`
  - spurious: `1: mechanical & electrical maintenance`
  - spurious: `2: inventory planning`
  - spurious: `2: data analysis`
  - spurious: `5.ocr: structural & construction works`
  - spurious: `6: alex morgan`
  - spurious: `6: backend & data systems`
- `header.phones`
  - missed: `5.ocr: +66 8x xxx xxxx`
- `section_routing`
  - missed: `5.ocr: others`
  - missed: `6: languages`
  - spurious: `1: licenses`
  - spurious: `7.anomaly: languages`
  - spurious: `7.anomaly: others`
- `experience.job_titles`
  - spurious: `5.ocr: eastern infrastructure services`
- `experience.companies`
  - missed: `5.ocr: eastern infrastructure services`
- `experience.locations`
  - missed: `9.ocr: jakarta, indonesia`
- `education.titles`
  - spurious: `7.anomaly: industrial mechanics / metal work (certificate level)`
- `education.institutions`
  - spurious: `7.anomaly: (fictional sample institution), bangkok`
- `education.entry_count`
  - spurious: `7.anomaly: entry-1`
