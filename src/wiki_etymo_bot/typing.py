from collections.abc import AsyncGenerator, Iterable
from typing import AsyncContextManager, Literal, Protocol

CacheKind = Literal["http", "llm"]

LlmPipelineName = Literal[
    "language_recognizer",
    "search_planner",
    "article_selector",
    "link_classifier",
    "response_synthesizer",
    "response_polisher",
    "reference_inserter",
]


class ProgressBar(Protocol):
    def __call__[T](
        self, iterable: Iterable[T], description: str, *, total: int | None = None
    ) -> AsyncGenerator[T]: ...


class ProgressSpinner(Protocol):
    def __call__(self, description: str) -> AsyncContextManager: ...
