from pathlib import Path
from typing import cast

import yaml
from aiohttp_client_cache.backends.sqlite import SQLiteBackend
from aiohttp_client_cache.session import CachedSession
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, TypeAdapter

from .context_formatter import ContextFormatter
from .core import DATA_DIR
from .output_models import (
    ArticleSelectorOutput,
    LanguageRecognizerOutput,
    LinkClassifierOutput,
    SearchPlannerOutput,
)
from .response_validator import ResponseValidator
from .settings import Settings
from .typing import LlmPipelineName
from .wiktionary_client import WiktionaryClient

CACHE_DIR = DATA_DIR / "cache"


class PromptTemplateMessages(BaseModel):
    system: str
    user: str


class PromptTemplateData(BaseModel):
    messages: PromptTemplateMessages
    history: bool = False


class Keywords(BaseModel):
    etymology: set[str]


class ResponseValidationConfig(BaseModel):
    negative_patterns: set[str]


class Resources:
    def __init__(self, settings: Settings):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.prompt_templates = self._load_yaml(
            path=settings.resources / "prompt_templates.yaml",
            model=dict[LlmPipelineName, PromptTemplateData],
        )
        self.keywords = self._load_yaml(
            path=settings.resources / "keywords.yaml",
            model=Keywords,
        )
        self.response_validation_config = self._load_yaml(
            path=settings.resources / "response_validation_config.yaml",
            model=ResponseValidationConfig,
        )
        self.embeddings = OllamaEmbeddings(
            model=settings.embeddings,
        )
        self.http_session = CachedSession(
            cache=None if "http" in settings.no_cache else SQLiteBackend(str(CACHE_DIR / "http_cache.sqlite"))
        )
        self.language_recognizer = self._build_llm_pipeline(settings, "language_recognizer", LanguageRecognizerOutput)
        self.search_planner = self._build_llm_pipeline(settings, "search_planner", SearchPlannerOutput)
        self.article_selector = self._build_llm_pipeline(settings, "article_selector", ArticleSelectorOutput)
        self.link_classifier = self._build_llm_pipeline(settings, "link_classifier", LinkClassifierOutput)
        self.response_synthesizer = self._build_llm_pipeline(settings, "response_synthesizer", str)
        self.response_polisher = self._build_llm_pipeline(settings, "response_polisher", str)
        self.reference_inserter = self._build_llm_pipeline(settings, "reference_inserter", str)
        self.context_formatter = ContextFormatter(max_article_length=settings.max_article_length)
        self.response_validator = ResponseValidator(negative_patterns=self.response_validation_config.negative_patterns)
        self.wiktionary_client = WiktionaryClient(
            settings=settings.wiktionary,
            http_session=self.http_session,
        )
        set_llm_cache(
            None if "llm" in settings.no_cache else SQLiteCache(database_path=str(CACHE_DIR / "llm_cache.sqlite"))
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.http_session.close()

    def _load_yaml[T](self, path: Path, model: type[T]) -> T:
        with open(path, "r", encoding="utf-8") as f:
            return TypeAdapter(model).validate_python(yaml.safe_load(f))

    def _build_llm_pipeline[T: BaseModel | str](
        self, settings: Settings, key: LlmPipelineName, output_model: type[T] = str
    ) -> Runnable[dict, T]:
        prompt_template_data = self.prompt_templates[key]
        message_templates = [
            ("system", prompt_template_data.messages.system),
            ("user", prompt_template_data.messages.user),
        ]
        if prompt_template_data.history:
            message_templates.insert(1, ("placeholder", "{history}"))
        prompt_template = ChatPromptTemplate(message_templates)
        if settings.reasoning == "off":
            reasoning = False
        else:
            reasoning = key in settings.reasoning
        llm = ChatOllama(
            model=settings.llm,
            temperature=0,
            validate_model_on_init=True,
            reasoning=reasoning,
        )
        if output_model is str:
            pipeline = prompt_template | llm | StrOutputParser()
        else:
            pipeline = prompt_template | llm.with_structured_output(output_model)
        return cast(Runnable[dict, T], pipeline)
