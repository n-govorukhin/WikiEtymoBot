import asyncio
import logging
import textwrap
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from operator import or_
from typing import Annotated, NamedTuple, cast
from uuid import uuid4

from iso639 import Lang
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph, add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from .article_retriever import ArticleRetriever
from .concurrency import BatchProcessor
from .core import Article, ArticleId, ArticleRequest, PageId
from .link_selector import LinkSelector
from .page_parser import PageParser
from .resources import Resources
from .settings import Settings
from .typing import ProgressBar, ProgressSpinner

logger = logging.getLogger("app")


class Response(NamedTuple):
    text: str
    references: str


class State(BaseModel, extra="forbid"):
    messages: Annotated[list[AnyMessage], add_messages] = []
    chat_language: str | None = None
    epoch: int = 0
    article_requests: set[ArticleRequest] = set()
    downloaded_pages: Annotated[dict[PageId, str], or_] = {}
    missing_page_ids: Annotated[set[PageId], or_] = set()
    retrieved_articles: Annotated[dict[ArticleRequest, Article], or_] = {}
    scanned_article_ids: Annotated[set[ArticleId], or_] = set()
    article_numbers: Annotated[dict[ArticleId, int], or_] = {}
    references: Annotated[dict[str, str], or_] = {}
    responses_with_inline_refs: Annotated[dict[str, str], or_] = {}

    @property
    def last_message(self) -> str:
        return cast(str, self.messages[-1].content)


class App:
    def __init__(
        self,
        *,
        settings: Settings,
        resources: Resources,
        progress_bar: ProgressBar,
        progress_spinner: ProgressSpinner,
    ):
        self._settings = settings
        self._resources = resources
        self._progress_bar = progress_bar
        self._progress_spinner = progress_spinner
        self._batch_processor = BatchProcessor(progress_bar)
        self._article_retriever = ArticleRetriever(
            page_parser=PageParser(
                embeddings=resources.embeddings,
                etymology_keywords=resources.keywords.etymology,
            ),
            article_selector=resources.article_selector,
            context_formatter=resources.context_formatter,
        )
        self._link_selector = LinkSelector(
            link_classifier=resources.link_classifier,
            batch_processor=self._batch_processor,
        )
        self._build_graph()

    async def send_message(self, message: str, thread_id: str = "default") -> Response:
        config = RunnableConfig({"configurable": {"thread_id": thread_id}})
        state = self._graph.get_state(config)
        if not state.interrupts:
            input = State(messages=[HumanMessage(message)])
        else:
            input = Command(resume=message)
        state = await self._graph.ainvoke(input, config=config)
        return state["__interrupt__"][0].value

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_sequence([self._recognize_language, self._request_articles, self._download_pages])
        graph.add_sequence([self._retrieve_articles, self._scan_articles])
        graph.add_sequence([self._insert_inline_refs, self._handle_io])
        graph.add_node(self._synthesize_response)
        graph.add_node(self._polish_response)
        graph.add_edge(START, "_recognize_language")
        graph.add_conditional_edges("_handle_io", self._route_after_handle_io)
        graph.add_conditional_edges("_download_pages", self._route_after_download_pages)
        graph.add_conditional_edges("_scan_articles", self._route_after_scan_articles)
        graph.add_conditional_edges("_synthesize_response", self._route_after_synthesize_response)
        graph.add_conditional_edges("_polish_response", self._route_after_polish_response)
        self._graph = graph.compile(
            checkpointer=InMemorySaver(
                serde=JsonPlusSerializer(
                    pickle_fallback=True,
                    allowed_msgpack_modules=[
                        (cls.__module__, cls.__qualname__) for cls in [ArticleId, ArticleRequest, PageId, Response]
                    ],
                ),
            ),
        )

    def _route_after_handle_io(self, state: State):
        if state.chat_language is None:
            return "_recognize_language"
        else:
            return "_request_articles"

    def _route_after_download_pages(self, state: State):
        if state.epoch <= self._settings.max_depth:
            return "_retrieve_articles"
        else:
            return "_synthesize_response"

    def _route_after_scan_articles(self, state: State):
        if state.article_requests:
            return "_download_pages"
        else:
            return "_synthesize_response"

    def _route_after_synthesize_response(self, state: State):
        if not self._resources.response_validator(response=state.last_message):
            logger.warning("Generated a misformatted response.")
            logger.debug(state.last_message)
            return "_polish_response"
        elif self._settings.inline_refs:
            return "_insert_inline_refs"
        else:
            return "_handle_io"

    def _route_after_polish_response(self, state: State):
        if self._settings.inline_refs:
            return "_insert_inline_refs"
        else:
            return "_handle_io"

    def _handle_io(self, state: State):
        assert (last_message_id := state.messages[-1].id)
        if self._settings.inline_refs:
            message = state.responses_with_inline_refs[last_message_id]
        else:
            message = state.last_message
        message = interrupt(Response(message, state.references[last_message_id]))
        return {"messages": [HumanMessage(message, id=str(uuid4()))]}

    async def _recognize_language(self, state: State):
        async with self._progress_spinner(self._format_progress_description("Recognizing language", state)):
            output = await self._resources.language_recognizer.ainvoke({"text": state.last_message})
        await asyncio.sleep(0.01)
        logger.info("Recognized the language: %s.", self._get_language_name(output.language))
        return {"chat_language": output.language}

    async def _request_articles(self, state: State):
        async with self._progress_spinner(self._format_progress_description("Planning search", state)):
            output = await self._resources.search_planner.ainvoke(
                {"question": state.last_message, "history": state.messages}
            )
        requests: set[ArticleRequest] = set()
        for word in output.words:
            request = ArticleRequest(
                page_id=PageId(subdomain=word.language_code, word=word.text),
                language=self._get_language_name(word.language_code),
                context=state.last_message,
            )
            requests.add(request)
            logger.info('Decided to search "%s" (%s).', word.text, request.language)
        assert state.chat_language is not None
        for request in list(requests):
            requests.update(self._augment_article_request(request, {"en", state.chat_language}))
        for request in requests:
            logger.info("Requested article %s (%s).", request.url, request.language)
        return {"article_requests": requests - state.retrieved_articles.keys()}

    async def _download_pages(self, state: State):
        downloaded_pages: dict[PageId, str] = {}
        errors: dict[PageId, Exception] = {}
        state.epoch += 1
        pending_page_ids = (
            {article_request.page_id for article_request in state.article_requests}
            - state.downloaded_pages.keys()
            - state.missing_page_ids
        )
        downloaded_pages, errors = await self._batch_processor(
            coros={page_id: self._resources.wiktionary_client.get(page_id) for page_id in pending_page_ids},
            description=self._format_progress_description("Downloading pages", state),
        )
        for page_id, error in errors.items():
            logger.error("Failed to load %s: %s.", page_id, error)
        for page_id in downloaded_pages:
            logger.debug("Downloaded page %s.", page_id.url)
        return {
            "downloaded_pages": downloaded_pages,
            "missing_page_ids": set(errors),
            "epoch": state.epoch,
        }

    async def _retrieve_articles(self, state: State):
        retrieved_articles, errors = await self._batch_processor(
            coros={
                request: self._article_retriever(state.downloaded_pages[request.page_id], request)
                for request in state.article_requests
                if request.page_id in state.downloaded_pages
            },
            description="Retrieving articles",
        )
        for request, error in errors.items():
            logger.error("Failed to retrieve article %s: %s.", request.url, error)
        for request, article in retrieved_articles.items():
            indent = 4
            text = "\n".join(" " * indent + line for line in textwrap.wrap(article.text, 120 - indent))
            logger.debug("Retrieved article %s:\n%s", article.id.url, text)
        return {"retrieved_articles": retrieved_articles}

    async def _scan_articles(self, state: State):
        assert state.chat_language is not None
        articles_to_scan = [
            state.retrieved_articles[request]
            for request in state.article_requests
            if request in state.retrieved_articles
            and state.retrieved_articles[request].id not in state.scanned_article_ids
        ]
        requests: set[ArticleRequest] = set()
        link_counter: Counter[ArticleId] = Counter()
        if article_links := await self._link_selector(
            articles=articles_to_scan,
            description=self._format_progress_description("Selecting links", state),
        ):
            padding = max(len(str(source.id.url)) for source, _ in article_links)
            for source, destination in article_links:
                if link_counter[source.id] == self._settings.max_links:
                    continue
                link_counter[source.id] += 1
                request = ArticleRequest(
                    page_id=destination.page_id,
                    language=destination.language or self._get_language_name(destination.page_id.subdomain),
                    context=source.text,
                )
                if self._settings.augment_links:
                    request_batch = self._augment_article_request(request, {"en", state.chat_language})
                else:
                    request_batch = [request]
                for request in request_batch:
                    if request not in state.retrieved_articles:
                        requests.add(request)
                        logger.info(f"Requested link %-{padding}s → %s.", source.id.url, request.url)
        assert state.chat_language is not None
        return {
            "scanned_article_ids": {article.id for article in articles_to_scan},
            "article_requests": requests,
        }

    async def _synthesize_response(self, state: State):
        article_numbers: dict[ArticleId, int] = {}
        references: list[str] = []
        unnumbered_article_ids = {
            article.id for article in state.retrieved_articles.values()
        } - state.article_numbers.keys()
        for article_id in sorted(unnumbered_article_ids):
            article_numbers[article_id] = article_number = len(state.article_numbers) + len(article_numbers) + 1
            references.append(f"[^{article_number}]: https://{article_id.url}")
        input = {
            "question": state.last_message,
            "articles": self._resources.context_formatter.format_articles(
                articles=set(state.retrieved_articles.values()),
                article_numbers=article_numbers | state.article_numbers,
            ),
            "history": state.messages[:-1],
        }
        state.epoch = 0
        async with self._progress_spinner(self._format_progress_description("Generating response", state)):
            response = await self._resources.response_synthesizer.ainvoke(input)
        response_message = AIMessage(response, id=str(uuid4()))
        return {
            "messages": [response_message],
            "epoch": state.epoch,
            "article_numbers": article_numbers,
            "references": {response_message.id: "\n".join(references)},
        }

    async def _polish_response(self, state: State):
        input = {
            "articles": self._resources.context_formatter.format_articles(
                articles=set(state.retrieved_articles.values()),
                article_numbers=state.article_numbers,
            ),
            "history": state.messages,
            "question": state.messages[-2].content,
            "response": state.messages[-1].content,
        }
        async with self._progress_spinner(self._format_progress_description("Polishing response", state)):
            response = await self._resources.response_polisher.ainvoke(input)
        return {
            "messages": [AIMessage(response, id=state.messages[-1].id)],
        }

    async def _insert_inline_refs(self, state: State):
        input = {
            "articles": self._resources.context_formatter.format_articles(
                articles=set(state.retrieved_articles.values()),
                article_numbers=state.article_numbers,
            ),
            "history": state.messages,
            "message": state.last_message,
        }
        async with self._progress_spinner(self._format_progress_description("Inserting references", state)):
            response = await self._resources.reference_inserter.ainvoke(input)
        return {
            "responses_with_inline_refs": {state.messages[-1].id: response},
        }

    def _augment_article_request(self, request: ArticleRequest, subdomains: set[str]) -> Iterable[ArticleRequest]:
        try:
            subdomain = Lang(request.language).pt1
        except Exception:
            pass
        else:
            subdomains = subdomains | {subdomain}
        for subdomain in subdomains:
            for apply_lowercase in [False, True]:
                page_id = replace(request.page_id, subdomain=subdomain)
                if apply_lowercase:
                    page_id = replace(page_id, word=page_id.word.lower())
                yield replace(request, page_id=page_id)

    def _get_language_name(self, language_code: str) -> str:
        try:
            return Lang(language_code).name
        except Exception:
            return language_code

    def _format_progress_description(self, description: str, state: State) -> str:
        return f"[Epoch {state.epoch}] {description}" if state.epoch > 0 else description
