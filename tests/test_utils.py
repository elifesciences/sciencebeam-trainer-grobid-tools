import logging
from functools import wraps
from typing import Dict, List


LOGGER = logging.getLogger(__name__)


def log_on_exception(f: callable) -> callable:
    """
    Wraps function to log error on exception.
    That is useful for tests that log a lot of things,
    and pytest displaying the test failure at the top of the method.
    (there doesn't seem to be an option to change that)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception('failed due to %s', repr(e))
            raise
    return wrapper


def dict_to_args(args_dict: Dict[str, str]) -> List[str]:
    return [
        '--%s' % key if isinstance(value, bool) and value else '--%s=%s' % (key, value)
        for key, value in args_dict.items()
        if value is not None
    ]
