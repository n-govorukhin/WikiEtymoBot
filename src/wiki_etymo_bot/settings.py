import sysconfig
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, NonNegativeFloat, PositiveInt
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliToggleFlag,
    SettingsConfigDict,
)

from .typing import CacheKind, LlmPipelineName


def get_default_resources_dir() -> Path:
    if (resources_dir := Path(__file__).resolve().parents[2] / "resources").exists():
        return resources_dir
    return Path(sysconfig.get_path("data")) / "share" / "wikietymobot" / "resources"


class WiktionaryClientSettings(BaseModel):
    concurrency_limit: PositiveInt = 1
    """ Максимальное количество одновременных запросов. """

    request_delay: NonNegativeFloat = 1
    """ Пауза в секундах, которую следует сделать после каждого запроса. """

    user_agent: str = "WikiEtymoBot/1.0 (test@example.com)"
    """ Значение заголовка User-Agent. """


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_avoid_json=True,
        cli_hide_none_type=True,
        cli_kebab_case=True,
        cli_prog_name="WikiEtymoBot",
        env_file=".env",
        env_nested_delimiter="_",
        env_prefix="wikietymobot_",
        extra="forbid",
        frozen=True,
        use_attribute_docstrings=True,
    )

    augment_links: CliToggleFlag[bool] = False
    """
    Запрашивать страницы из ссылок на дополнительных языках, включая английский язык, язык, к которому относится слово,
    и язык, на котором задан вопрос.
    """

    cli_color: str = "cyan"
    """ Основной цвет интерфейса консольного приложения. """

    embeddings: str = Field(default="embeddinggemma:300m")
    """ Название модели для векторного представления текста, используемой через Ollama. """

    inline_refs: CliToggleFlag[bool] = False
    """ Вставлять в ответ сноски на статьи после каждого утверждения. """

    llm: str = Field(default="qwen3:14b")
    """ Название LLM, используемой через Ollama. """

    log_level: Annotated[str, BeforeValidator(str.upper)] = "CRITICAL"
    """ Уровень логирования. """

    max_article_length: int = Field(default=5000, ge=100)
    """ Максимальная длина статьи, которая попадает в контекст LLM. Более длинные статьи обрезаются. """

    max_depth: PositiveInt = 7
    """ Максимальная длина пути переходов по ссылкам. """

    max_links: PositiveInt = 5
    """ Максимальное количество ссылок, на которые можно перейти из одной статьи. """

    message: CliPositionalArg[str | None] = None
    """
    Входное сообщение.
    Если указано, то приложение даёт ответ на него и завершает работу.
    Если не указано, то приложение запускается в режиме чата.
    """

    no_cache: set[CacheKind] = set()
    """ Компоненты, для которых следует отключить кеширование данных. """

    reasoning: list[LlmPipelineName] | Literal["off"] = ["response_synthesizer"]
    """
    Шаги алгоритма, на которых следует включить режим размышлений в LLM.
    Значение "off" используется для отключения размышлений на всех шагах.
    """

    resources: Path = Field(default_factory=get_default_resources_dir)
    """ Путь к папке со статическими данными. """

    version: CliToggleFlag[bool] = False
    """ Показать версию приложения. """

    wiktionary: WiktionaryClientSettings = WiktionaryClientSettings()
    """ Настройки клиента для API Викисловаря. """
