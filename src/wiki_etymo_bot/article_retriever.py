from bs4 import BeautifulSoup
from langchain_core.runnables import Runnable

from .context_formatter import ContextFormatter
from .core import Article, ArticleRequest
from .output_models import ArticleSelectorOutput
from .page_parser import PageParser


class ArticleRetriever:
    def __init__(
        self,
        *,
        page_parser: PageParser,
        article_selector: Runnable[dict, ArticleSelectorOutput],
        context_formatter: ContextFormatter,
    ):
        self._page_parser = page_parser
        self._article_selector = article_selector
        self._context_formatter = context_formatter

    async def __call__(self, page: str, request: ArticleRequest) -> Article:
        soup = BeautifulSoup(page, "html.parser")
        articles = self._page_parser(soup, request.page_id, request.language)
        if not articles:
            raise ValueError("etymology section not found")
        if len(articles) == 1:
            return articles[0]
        input = {
            "word": request.page_id.word,
            "context": request.context,
            "articles": self._context_formatter.format_articles(
                articles=articles,
                article_numbers={article.id: number for number, article in enumerate(articles, 1)},
            ),
        }
        output = await self._article_selector.ainvoke(input)
        return articles[output.article_id - 1]
