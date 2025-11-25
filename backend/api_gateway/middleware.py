"""
FastAPI middleware for logging, CORS, and error handling
"""

import time
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response details
        """
        request_id = request.headers.get("X-Request-ID", f"{time.time()}")
        start_time = time.time()

        # Log request
        logger.info(
            f"{request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"Response {response.status_code}",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=int(duration * 1000),
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(duration)

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {str(e)}",
                request_id=request_id,
                exception=str(e),
                traceback=traceback.format_exc(),
                duration_ms=int(duration * 1000),
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global error handling
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Catch and format errors
        """
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(
                f"Unhandled exception: {str(e)}",
                exception=str(e),
                traceback=traceback.format_exc(),
                path=request.url.path,
            )

            if settings.DEBUG:
                detail = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            else:
                detail = {"error": "Internal server error"}

            return JSONResponse(
                status_code=500,
                content=detail,
            )


def setup_cors(app):
    """
    Configure CORS middleware

    Args:
        app: FastAPI application
    """
    origins = settings.CORS_ORIGINS.split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    logger.info(f"CORS configured for origins: {origins}")


def setup_middleware(app):
    """
    Configure all middleware

    Args:
        app: FastAPI application
    """
    # Add error handling first
    app.add_middleware(ErrorHandlingMiddleware)

    # Add logging
    app.add_middleware(LoggingMiddleware)

    # Setup CORS
    setup_cors(app)

    logger.info("Middleware configured")
