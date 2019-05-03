def comma_separated_str_to_list(s):
    s = s.strip()
    if not s:
        return []
    return [item.strip() for item in s.split(',')]
