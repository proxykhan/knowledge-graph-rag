"""Read cached extraction results for the configured model and prompt version."""

import json
from collections.abc import Iterator

from kgrag import llm
from kgrag.config import CACHE_DIR
from kgrag.extract.prompt import PROMPT_VERSION


def load_extractions() -> Iterator[dict]:
    root = CACHE_DIR / "extraction" / llm.model_name() / PROMPT_VERSION
    for path in sorted(root.glob("*/*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))
