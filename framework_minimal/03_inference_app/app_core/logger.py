import logging
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def define_log_level(print_level: str = "INFO", logfile_level: str = "DEBUG", name: str | None = None):
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{formatted_date}" if name else formatted_date

    log_path = PROJECT_ROOT / "logs"
    log_path.mkdir(exist_ok=True)

    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level=print_level)
        _logger.add(log_path / f"{log_name}.log", level=logfile_level)
        return _logger
    except ImportError:
        logger = logging.getLogger("robotplanner")
        logger.setLevel(getattr(logging, logfile_level.upper(), logging.DEBUG))
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(getattr(logging, print_level.upper(), logging.INFO))
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_path / f"{log_name}.log", encoding="utf-8")
        file_handler.setLevel(getattr(logging, logfile_level.upper(), logging.DEBUG))
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger


logger = define_log_level()
