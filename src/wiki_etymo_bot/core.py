import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Self

DATA_DIR = Path.home() / ".wikietymobot"


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PageId:
    subdomain: str
    word: str

    @property
    def url(self) -> "PageUrl":
        return PageUrl(page_id=self)

    def __str__(self):
        return str(self.url)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PageUrl:
    page_id: PageId
    language: str | None = None

    @classmethod
    def from_string(cls, s: str) -> Self | None:
        if match := re.match(r"(?:https://)?(\w+)\.wiktionary\.org/wiki/([^#]+)(?=#(.+))?", s):
            return cls(
                page_id=PageId(subdomain=match.group(1), word=match.group(2)),
                language=match.group(3),
            )

    def __str__(self):
        result = f"{self.page_id.subdomain}.wiktionary.org/wiki/{self.page_id.word}"
        if self.language is not None:
            result += "#" + self.language
        return result


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ArticleKeyBase:
    page_id: PageId
    language: str

    @property
    def url(self) -> PageUrl:
        return PageUrl(page_id=self.page_id, language=self.language.replace(" ", "_"))


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ArticleId(ArticleKeyBase):
    heading: str


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ArticleRequest(ArticleKeyBase):
    context: str


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class Article:
    id: ArticleId
    text: str = field(hash=False)


class ArticleLink(NamedTuple):
    source: Article
    destination: PageUrl
