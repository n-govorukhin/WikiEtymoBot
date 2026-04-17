import textwrap
from collections.abc import Iterable
from xml.etree import ElementTree as ET

from .core import Article, ArticleId


class ContextFormatter:
    def __init__(self, max_article_length: int | None = None):
        self._max_article_length = max_article_length

    def format_article(self, article: Article, **extra_attrs) -> str:
        attrs = extra_attrs | {"word": article.id.page_id.word, "language": article.id.language}
        element = ET.Element("article", attrib=attrs)
        element.text = self._crop_article(article.text)
        return ET.tostring(element, encoding="unicode")

    def format_articles(self, articles: Iterable[Article], article_numbers: dict[ArticleId, int]) -> str:
        articles = sorted(articles, key=lambda article: article_numbers[article.id])
        if not articles:
            return "[NO ARTICLES FOUND]"
        return "\n".join(self.format_article(article, id=str(article_numbers[article.id])) for article in articles)

    def _crop_article(self, text: str) -> str:
        if self._max_article_length is None:
            return text
        return textwrap.shorten(text, self._max_article_length, placeholder="... [CONTENT CROPPED]")
