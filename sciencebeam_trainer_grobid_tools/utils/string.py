from typing import Dict, Tuple


def is_blank(text: str) -> bool:
    return not text or not text.strip()


def comma_separated_str_to_list(s):
    s = s.strip()
    if not s:
        return []
    return [item.strip() for item in s.split(',')]


def parse_key_value(expr: str) -> Tuple[str, str]:
    key, value = expr.split('=', maxsplit=1)
    return key.strip(), value.strip()


def parse_dict(expr: str, delimiter: str = '|') -> Dict[str, str]:
    if not expr:
        return {}
    d = {}
    for fragment in expr.split(delimiter):
        key, value = parse_key_value(fragment)
        d[key] = value
    return d
