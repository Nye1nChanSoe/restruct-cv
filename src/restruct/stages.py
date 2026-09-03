"""Which debug artifacts each stage owns.

Its own module so the CLI can read it without importing the pipeline, which
pulls in torch and transformers and costs seconds. A run that fails validation,
or asks for --help, should not pay for a model library it never touches.
"""

from __future__ import annotations

# Selecting stages never decides whether a pass runs. The whole pipeline always
# executes, because every pass feeds the next and the result would otherwise be
# a different result rather than a less-documented one.
ALL_STAGES: frozenset[int] = frozenset({1, 2, 3, 4, 5})
DEFAULT_DEBUG_STAGES: frozenset[int] = frozenset({4, 5})
