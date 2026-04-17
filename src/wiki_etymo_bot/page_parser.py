from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import dropwhile, islice
from urllib.parse import unquote as unquote_url

import html_to_markdown
from bs4 import BeautifulSoup, Tag
from bs4.element import PageElement
from langchain_core.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity

from .core import Article, ArticleId, PageId


@dataclass(slots=True)
class SectionElement:
    value: PageElement
    level: int
    section_heading: str
    is_section_start: bool = False


class PageParser:
    def __init__(self, etymology_keywords: set[str], embeddings: Embeddings):
        self._etymology_keywords = etymology_keywords
        self._embeddings = embeddings

    def __call__(self, soup: BeautifulSoup, page_id: PageId, language: str) -> list[Article]:
        language, language_section = self._extract_language_section(soup, language)
        etymologies = list(self._extract_etymologies(language_section, page_id, language))
        if not etymologies:
            base_level = self._get_min_heading_level(language_section)
            if base_level is None:
                raise ValueError("subsections not found")
            for heading in self._get_section_headings(language_section, base_level):
                section = list(self._filter_section_elements(language_section, base_level, heading))
                etymologies.extend(self._extract_etymologies(section, page_id, language))
        return etymologies

    def _extract_language_section(self, soup: BeautifulSoup, language: str) -> tuple[str, list[SectionElement]]:
        elements = list(self._get_section_elements(soup))
        base_level = self._get_min_heading_level(elements)
        if base_level is None:
            raise ValueError("sections not found")
        match list(self._get_section_headings(elements, base_level)):
            case [language]:
                pass
            case languages:
                assert len(languages) > 0
                language_index = self._choose_semantically_closest(language, languages)
                language = languages[language_index]
        return self._clean_heading(language), list(self._filter_section_elements(elements, base_level, language))

    def _extract_etymologies(self, section: list[SectionElement], page_id: PageId, language: str) -> Iterable[Article]:
        base_level = self._get_min_heading_level(section)
        if base_level is None:
            return {}
        headings = list(self._get_section_headings(section, base_level))
        lowercase_headings = list(map(str.lower, headings))
        etymology_headings = (
            heading
            for heading, lowercase_heading in zip(headings, lowercase_headings)
            if any(keyword in lowercase_heading for keyword in self._etymology_keywords)
        )
        for heading in etymology_headings:
            etymology_section = list(self._filter_section_elements(section, base_level, heading))
            if etymology := self._compose_etymology_section(etymology_section, page_id):
                yield Article(
                    id=ArticleId(page_id=page_id, language=language, heading=self._clean_heading(heading)),
                    text=etymology,
                )

    def _compose_etymology_section(self, section: list[SectionElement], page_id: PageId) -> str | None:
        html = ""
        for element in section:
            if isinstance(element.value, Tag) and element.value.name == "p":
                paragraph = deepcopy(element.value)
                self._clean_html_tag(paragraph, page_id)
                html += str(paragraph)
        if etymology := html_to_markdown.convert(html).content:
            etymology = etymology.replace("\xa0", " ")
        return etymology

    def _get_section_elements(self, soup: BeautifulSoup) -> Iterable[SectionElement]:
        level = 0
        section_heading = None
        for element in soup.find_all():
            if (
                isinstance(element, Tag)
                and element.name == "div"
                and element.has_attr("class")
                and element["class"][0] == "mw-heading"
            ):
                is_section_start = True
                level = int(element["class"][1][len("mw-heading") :])
                assert isinstance(heading_tag := next(element.children), Tag)
                section_heading = str(heading_tag["id"])
            else:
                is_section_start = False
            if level != 0:
                assert section_heading is not None
                yield SectionElement(element, level, section_heading, is_section_start)

    def _clean_html_tag(self, tag: Tag, page_id: PageId) -> None:
        for tag in tag.find_all_next():
            if tag.name == "a":
                if tag.has_attr("href"):
                    href = str(tag["href"])
                    if (
                        href.startswith("/wiki")  # ссылка на викисловарь
                        and ":" not in href  # не ссылка на категорию
                        and not tag.text.endswith(".")  # не ссылка на аббревиатуру
                    ):
                        # Нормализуем ссылки на Викисловарь.
                        tag["href"] = f"{page_id.subdomain}.wiktionary.org" + unquote_url(href)
                    else:
                        # Убираем ссылки, которые ведут не на Викисловарь.
                        tag.unwrap()
                if tag.has_attr("title"):
                    # Убираем название ссылки.
                    del tag["title"]
            elif tag.name in ("i", "b", "strong", "mark", "del"):
                # Убираем выделения текста.
                tag.unwrap()
            elif tag.name in ("sub", "sup"):
                # Убираем верхние и нижние индексы.
                tag.decompose()

    def _clean_heading(self, heading: str) -> str:
        return heading.replace("_", " ")

    def _filter_section_elements(
        self, elements: Iterable[SectionElement], level: int, heading: str
    ) -> Iterable[SectionElement]:
        skip_condition = lambda element: not (element.level == level and element.section_heading == heading)
        for element in islice(dropwhile(skip_condition, elements), 1, None):
            if element.is_section_start and element.level <= level:
                break
            yield element

    def _get_min_heading_level(self, elements: Iterable[SectionElement]) -> int | None:
        return min((element.level for element in elements if element.is_section_start), default=None)

    def _get_section_headings(self, elements: Iterable[SectionElement], level: int) -> Iterable[str]:
        for element in elements:
            if isinstance(element.value, Tag) and element.value.name == f"h{level}":
                yield element.section_heading

    def _choose_semantically_closest(self, query: str, choices: list[str]) -> int:
        if query in choices:
            return choices.index(query)
        embeddings = self._embeddings.embed_documents(choices + [query])
        return cosine_similarity(embeddings[:-1], [embeddings[-1]]).argmax()  # type: ignore
