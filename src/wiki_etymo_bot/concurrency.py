import asyncio
from typing import Any, Coroutine, cast

from .typing import ProgressBar


class BatchProcessor:
    def __init__(self, progress_bar: ProgressBar):
        self._progress_bar = progress_bar

    async def __call__[K, V](
        self, coros: dict[K, Coroutine[Any, Any, V]], description: str
    ) -> tuple[dict[K, V], dict[K, Exception]]:
        results: dict[K, V] = {}
        exceptions: dict[K, Exception] = {}
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._wrap(key, coro)) for key, coro in coros.items()]
            async for future in self._progress_bar(asyncio.as_completed(tasks), description, total=len(tasks)):
                key, result, exception = await future
                if exception is None:
                    results[key] = cast(V, result)
                else:
                    exceptions[key] = exception
        return results, exceptions

    async def _wrap[K, V](self, key: K, coro: Coroutine[Any, Any, V]) -> tuple[K, V | None, Exception | None]:
        try:
            return key, (await coro), None
        except Exception as exception:
            return key, None, exception
