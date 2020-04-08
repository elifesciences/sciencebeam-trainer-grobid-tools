def is_blank(text: str) -> bool:
    return not text or not text.strip()


def comma_separated_str_to_list(s):
    s = s.strip()
    if not s:
        return []
    return [item.strip() for item in s.split(',')]
