import re
from collections.abc import Iterable

from langchain_core.runnables import Runnable

from .concurrency import BatchProcessor
from .core import Article, ArticleLink, PageUrl
from .output_models import LinkClassifierOutput


class LinkSelector:
    def __init__(self, link_classifier: Runnable[dict, LinkClassifierOutput], batch_processor: BatchProcessor):
        self._link_classifier = link_classifier
        self._batch_processor = batch_processor

    def _extract_quoted_texts(self, text: str) -> list[str]:
        patterns = re.compile(r"«(.+?)»"), re.compile(r"“(.+?)”")
        return [result for pattern in patterns for result in pattern.findall(text)]

    def _unfold_links(self, text: str) -> str:
        return re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    def _unfold_quoted_links(self, text: str) -> str:
        for quoted_text in self._extract_quoted_texts(text):
            text = text.replace(quoted_text, self._unfold_links(quoted_text))
        return text

    def _extract_links(self, articles: Iterable[Article]) -> Iterable[ArticleLink]:
        for article in articles:
            article_text = self._unfold_quoted_links(article.text)
            for url in set(re.findall(r"\[.+?\]\((.+?)\)", article_text)):
                if page_url := PageUrl.from_string(url):
                    yield ArticleLink(source=article, destination=page_url)

    async def __call__(self, articles: Iterable[Article], description: str) -> list[ArticleLink]:
        outputs, _ = await self._batch_processor(
            coros={
                link: self._link_classifier.ainvoke(
                    input={
                        "word": link.source.id.page_id.word,
                        "article": link.source.text,
                        "link": link.destination,
                        "link_word": link.destination.page_id.word,
                    }
                )
                for link in set(self._extract_links(articles))
            },
            description=description,
        )
        return [link for link, output in outputs.items() if output.answer == 1]
