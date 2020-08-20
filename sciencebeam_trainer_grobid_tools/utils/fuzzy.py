import logging
from difflib import SequenceMatcher
from typing import Tuple

from sciencebeam_alignment.align import (
    LocalSequenceMatcher
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    FuzzyMatchResult,
    DEFAULT_SCORING
)


LOGGER = logging.getLogger(__name__)


DEFAULT_WORD_SEPARATORS = ' .,-:()[]'


def DEFAULT_ISJUNK(s, i):
    return (
        (i > 0 and s[i - 1] == '.' and (s[i] == ' ' or s[i] == ','))
        or (i > 0 and s[i - 1].isalpha() and s[i] == '.')
        or (i > 0 and s[i - 1] == s[i])
        or s[i] == '*'
        or s[i] == ' '
    )


def split_with_offset(s: str, sep: str, include_separators: bool = True):
    previous_start = 0
    tokens = []
    for i, c in enumerate(s):
        if c in sep:
            if previous_start < i:
                tokens.append((previous_start, s[previous_start:i]))
            if include_separators:
                tokens.append((i, c))
            previous_start = i + 1
    if previous_start < len(s):
        tokens.append((previous_start, s[previous_start:]))
    return tokens


class WordSequenceMatcher(object):
    def __init__(self, isjunk=None, a=None, b=None, sep=None):
        if isjunk:
            raise ValueError('isjunk not supported')
        self.a = a
        self.b = b
        self.sep = sep or DEFAULT_WORD_SEPARATORS

    def get_matching_blocks(self):
        a_words_with_offsets = split_with_offset(self.a, self.sep)
        b_words_with_offsets = split_with_offset(self.b, self.sep)
        a_words = [w for _, w in a_words_with_offsets]
        b_words = [w for _, w in b_words_with_offsets]
        a_indices = [i for i, _ in a_words_with_offsets]
        b_indices = [i for i, _ in b_words_with_offsets]
        sm = SequenceMatcher(None, a_words, b_words, autojunk=False)
        raw_matching_blocks = sm.get_matching_blocks()
        matching_blocks = [
            (
                a_indices[ai],
                b_indices[bi],
                sum(
                    len(a_words[ai + token_index])
                    for token_index in range(size)
                )
            )
            for ai, bi, size in raw_matching_blocks
            if size
        ]
        return matching_blocks


def fuzzy_search(
        haystack: str, needle: str,
        threshold: float,
        exact_word_match_threshold: int = 5,
        start_index: int = 0,
        isjunk: callable = None) -> FuzzyMatchResult:
    original_haystack = haystack
    if start_index:
        haystack = haystack[start_index:]
    if min(len(haystack), len(needle)) < exact_word_match_threshold:
        sm = WordSequenceMatcher(None, haystack, needle, sep=DEFAULT_WORD_SEPARATORS)
    else:
        sm = LocalSequenceMatcher(a=haystack, b=needle, scoring=DEFAULT_SCORING)
    matching_blocks = sm.get_matching_blocks()
    if start_index:
        matching_blocks = [
            (ai + start_index, bi, size)
            for ai, bi, size in matching_blocks
        ]
    fm = FuzzyMatchResult(
        original_haystack,
        needle,
        matching_blocks,
        isjunk=isjunk or DEFAULT_ISJUNK
    )
    LOGGER.debug('fm (sm=%s, threshold=%.2f): %s', type(sm).__name__, threshold, fm)
    if fm.b_gap_ratio() >= threshold:
        return fm
    return None


def iter_fuzzy_search_all(haystack: str, *args, start_index: int = 0, **kwargs) -> FuzzyMatchResult:
    while start_index < len(haystack):
        fm = fuzzy_search(haystack, *args, **kwargs, start_index=start_index)
        if not fm:
            return
        yield fm
        new_start_index = fm.a_index_range()[1]
        if new_start_index <= start_index:
            return
        start_index = new_start_index


def fuzzy_search_index_range(*args, **kwargs) -> Tuple[int, int]:
    fm = fuzzy_search(*args, **kwargs)
    if fm:
        return fm.a_index_range()
    return None


def iter_fuzzy_search_all_index_ranges(*args, **kwargs) -> Tuple[int, int]:
    return (
        fm.a_index_range()
        for fm in iter_fuzzy_search_all(*args, **kwargs)
    )
