from halo import Halo

loading_spinner = Halo("Loading", placement="right").start()

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from importlib.metadata import version

from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style
from pydantic_settings import CliApp
from termcolor import colored
from tqdm import tqdm

from wiki_etymo_bot.app import App
from wiki_etymo_bot.core import DATA_DIR
from wiki_etymo_bot.resources import Resources
from wiki_etymo_bot.settings import Settings

PROGRESS_RENDER_DELAY = 0.01

settings = CliApp.run(Settings, cli_exit_on_error=False)
if settings.version:
    loading_spinner.stop()
    print("WikiEtymoBot", version("wikietymobot"))
    exit(0)


class ColoredLogFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "grey",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    def format(self, record):
        return colored(super().format(record), self.COLORS.get(record.levelno))


@asynccontextmanager
async def progress_spinner_context_manager(description: str):
    with Halo(description, color=settings.cli_color, placement="right") as halo:
        yield halo
    await asyncio.sleep(PROGRESS_RENDER_DELAY)


async def progress_bar_generator[T](
    iterable: Iterable[T], description: str, *, total: int | None = None
) -> AsyncGenerator[T]:
    for element in tqdm(iterable, desc=description, total=total, leave=False):
        yield element
    await asyncio.sleep(PROGRESS_RENDER_DELAY)


def configure_logging():
    logger = logging.getLogger("chat")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(settings.log_level)
    handler.setFormatter(ColoredLogFormatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)


def print_response(response: str, references: str) -> None:
    if references:
        response += "\n\n" + references
    print(f"\n{response}\n")


async def run():
    async with Resources(settings) as resources:
        app = App(
            settings=settings,
            resources=resources,
            progress_bar=progress_bar_generator,
            progress_spinner=progress_spinner_context_manager,
        )
        loading_spinner.stop()
        if sys.stdin.isatty() and settings.message is None:
            prompt_session = PromptSession(
                message="> ",
                style=Style.from_dict({"": settings.cli_color}),
                history=FileHistory(str(DATA_DIR / "input_history.txt")),
            )
            while (message := await prompt_session.prompt_async()) != "/exit":
                response, references = await app.send_message(message)
                print_response(response, references)
        else:
            if settings.message is not None:
                message = settings.message
            else:
                message = str.strip(sys.stdin.buffer.read().decode())
            response, references = await app.send_message(message)
            print_response(response, references)


def main() -> None:
    configure_logging()
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
        raise SystemExit(0)


if __name__ == "__main__":
    main()
