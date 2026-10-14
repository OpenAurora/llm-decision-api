import argparse
import logging

import uvicorn
from pydantic import ValidationError

from .api import create_app
from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="Run the LLM Decision HTTP API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8010, help="Bind port (default: %(default)s)")
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
        access_log=False,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
