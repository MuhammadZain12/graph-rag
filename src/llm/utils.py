import time
import functools
import random
from typing import Callable, Any


def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 40.0,
    jitter: bool = True,
):
    """
    A manual retry decorator with exponential backoff.
    """

    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i == retries:
                        print(f"FAILED: All {retries} retries exhausted.")
                        raise last_exception

                    # Log the error
                    print(f"ERROR: {str(e)[:100]}... (Attempt {i+1}/{retries+1})")

                    # Calculate sleep time
                    sleep_time = delay
                    if jitter:
                        sleep_time += random.uniform(0, 0.1 * delay)

                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

                    # Update delay for next round
                    delay *= backoff_factor

            return None  # Should not reach here

        return wrapper

    return decorator
