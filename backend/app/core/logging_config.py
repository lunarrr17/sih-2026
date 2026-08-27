import logging
import sys
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class ColoredFormatter(logging.Formatter):
    """Custom colorized console log formatter for Docker stdout visualization."""
    
    GREY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD + RED + FORMAT + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logging():
    """Configures global application logging with colorized console output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    
    # Set uvicorn and app loggers
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "backend"]:
        l = logging.getLogger(name)
        l.handlers = [handler]
        l.propagate = False

logger = logging.getLogger("backend.pipeline")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests with response status and execution latency."""
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        logger.info(f"📥 [{method}] {path} - Request Received")
        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"📤 [{method}] {path} -> {response.status_code} ({duration_ms}ms)")
            return response
        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"❌ [{method}] {path} -> FAILED ({duration_ms}ms): {exc}")
            raise exc
