import logging
import os


ENV_MODE = os.getenv("ENV_MODE", "LOCAL")


try:
    import structlog

    renderer = [structlog.processors.JSONRenderer()]
    if ENV_MODE.lower() == "local":
        renderer = [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.dict_tracebacks,
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.contextvars.merge_contextvars,
            *renderer,
        ],
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(level=logging.DEBUG)
except ImportError:
    logger = logging.getLogger("robotplanner.utils")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
