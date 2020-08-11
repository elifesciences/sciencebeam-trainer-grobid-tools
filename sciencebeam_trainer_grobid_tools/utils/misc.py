from typing import List, T


def get_safe(a: List[T], key: int, default_value: T = None) -> T:
    try:
        return a[key]
    except (IndexError, TypeError):
        return default_value
