from typing import Dict, Optional, Sequence, TypeVar


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def get_dict_safe(
    d: Optional[Dict[K, V]],
    key: Optional[K],
    default_value: Optional[V] = None
) -> Optional[V]:
    if d is None or key is None:
        return default_value
    return d.get(key, default_value)


def get_safe(a: Optional[Sequence[T]], key: int, default_value: Optional[T] = None) -> Optional[T]:
    try:
        return a[key]
    except (IndexError, TypeError):
        return default_value
