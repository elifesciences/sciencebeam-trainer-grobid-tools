from typing import Optional, Sequence, T


def get_safe(a: Optional[Sequence[T]], key: int, default_value: Optional[T] = None) -> Optional[T]:
    try:
        return a[key]
    except (IndexError, TypeError):
        return default_value
