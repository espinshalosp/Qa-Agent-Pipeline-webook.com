"""
utils/retry.py
==============
Exponential-backoff retry decorator for HTTP calls.

Usage
-----
    from utils.retry import retry_on_failure

    @retry_on_failure(max_retries=3, backoff_factor=2, retry_codes={429, 503})
    def call_api():
        ...

The decorator handles:
  - httpx.HTTPStatusError  — checks status code vs retry_codes
  - httpx.RequestError     — always retried (network-level failures)
  - Fatal codes            — [400, 401, 403, 404] are NOT retried; re-raised immediately
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Set

from utils.logger import get_logger

log = get_logger(__name__)

# Status codes that indicate a programming / auth error — never retry these
FATAL_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404})


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: int = 2,
    retry_codes: Set[int] | None = None,
) -> Callable:
    """
    Decorator factory that wraps a function with retry logic.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts after the first failure.
    initial_delay : float
        Seconds to wait before the first retry.
    backoff_factor : int
        Multiplier applied to the delay on each subsequent retry.
    retry_codes : set[int] | None
        HTTP status codes that should trigger a retry. Defaults to {429, 503}.
    """
    if retry_codes is None:
        retry_codes = {429, 503}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exc: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as exc:
                    # Determine whether we can import httpx to inspect the error
                    try:
                        import httpx
                        is_http_status_error = isinstance(exc, httpx.HTTPStatusError)
                        is_request_error = isinstance(exc, httpx.RequestError)
                    except ImportError:
                        is_http_status_error = False
                        is_request_error = False

                    if is_http_status_error:
                        status_code: int = exc.response.status_code  # type: ignore[attr-defined]

                        # Fatal — never retry
                        if status_code in FATAL_STATUS_CODES:
                            log.error(
                                "Fatal HTTP error — aborting retry. "
                                f"Status {status_code} from {exc.request.url}"  # type: ignore[attr-defined]
                            )
                            raise

                        # Retryable
                        if status_code in retry_codes and attempt < max_retries:
                            log.warning(
                                f"HTTP {status_code} received. "
                                f"Retrying (attempt {attempt + 1}/{max_retries}) in {delay:.1f}s..."
                            )
                            last_exc = exc
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue

                        # Non-retryable, non-fatal status — re-raise immediately
                        raise

                    elif is_request_error and attempt < max_retries:
                        # Network-level errors are always retried
                        log.warning(
                            f"Network error: {exc}. "
                            f"Retrying (attempt {attempt + 1}/{max_retries}) in {delay:.1f}s..."
                        )
                        last_exc = exc
                        time.sleep(delay)
                        delay *= backoff_factor
                        continue

                    else:
                        # Unknown error or last attempt — re-raise
                        raise

            # Exhausted all retries
            log.error(f"All {max_retries} retry attempts exhausted.")
            if last_exc is not None:
                raise last_exc

        return wrapper

    return decorator
