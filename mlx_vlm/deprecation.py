import functools
import warnings
from typing import Callable, Optional


def deprecate(
    remove_version: str,
    message: str,
    instead: Optional[str] = None,
    since: Optional[str] = None,
) -> Callable:
    """
    Mark a function or method as deprecated.

    Args:
        remove_version: Version when this will be removed
        message: Deprecation message
        instead: What to use instead
        since: Version when this was deprecated

    Example:
        @deprecate(
            remove_version="2.0.0",
            message="Legacy API function",
            instead="new_api()",
            since="1.0.0"
        )
        def old_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"`{func.__name__}` is deprecated"

            if since:
                msg += f" since v{since}"

            msg += f". {message}"

            if instead:
                msg += f" Use `{instead}` instead."

            msg += f" Will be removed in v{remove_version}."

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
