import logging
import inspect
from functools import wraps
from typing import List, Mapping, Optional, Union


LOGGER = logging.getLogger(__name__)


def wrap_class_methods(cls, wrapper):
    for key, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, key, wrapper(value))
    return cls


def log_on_exception(f: Union[callable, type]) -> callable:
    """
    Wraps function to log error on exception.
    That is useful for tests that log a lot of things,
    and pytest displaying the test failure at the top of the method.
    (there doesn't seem to be an option to change that)
    """
    if inspect.isclass(f):
        return wrap_class_methods(f, log_on_exception)

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception('failed due to %s', repr(e))
            raise
    return wrapper


def dict_to_args(args_dict: Mapping[str, Optional[Union[str, bool, int]]]) -> List[str]:
    return [
        (
            '--%s' % key if isinstance(value, bool) and value
            else '--%s=%s' % (key, value)
        )
        for key, value in args_dict.items()
        if (
            value is not None
            and (not isinstance(value, bool) or value)
        )
    ]
