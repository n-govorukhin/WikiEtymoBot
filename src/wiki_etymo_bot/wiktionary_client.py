import asyncio

from aiohttp_client_cache.session import CachedSession

from .core import PageId
from .settings import WiktionaryClientSettings


class WiktionaryClient:
    def __init__(self, http_session: CachedSession, settings: WiktionaryClientSettings):
        self._http_session = http_session
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.concurrency_limit)

    async def get(self, page_id: PageId) -> str:
        async with self._semaphore:
            async with self._http_session.get(
                url=f"https://{page_id.subdomain}.wiktionary.org/w/api.php",
                params={
                    "action": "parse",
                    "format": "json",
                    "prop": "text",
                    "page": page_id.word,
                },
                headers={"User-Agent": self._settings.user_agent},
            ) as response:
                response.raise_for_status()
                if not getattr(response, "from_cache"):
                    await asyncio.sleep(self._settings.request_delay)
                data = await response.json()
                try:
                    return data["parse"]["text"]["*"]
                except KeyError:
                    raise ValueError("the page is empty")
