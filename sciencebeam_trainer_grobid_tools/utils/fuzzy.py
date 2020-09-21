import logging
from difflib import SequenceMatcher
from typing import List, Tuple

from sciencebeam_alignment.align import (
    LocalSequenceMatcher
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    FuzzyMatchResult,
    DEFAULT_SCORING
)


LOGGER = logging.getLogger(__name__)


DEFAULT_WORD_SEPARATORS = ' .,-:;()[]'

NOT_SET = 'NOT_SET'


def default_is_junk(s, i):
    ch = s[i]
    if ch in {'*', ' '}:
        return True
    prev_non_space_index = i - 1
    while prev_non_space_index >= 0 and s[prev_non_space_index] == ' ':
        prev_non_space_index -= 1
    prev_non_space_ch = s[prev_non_space_index] if prev_non_space_index >= 0 else ''
    if ch == ',' and prev_non_space_ch == '.':
        return True
    if ch == '.' and prev_non_space_ch.isalpha():
        return True
    return False


def space_is_junk(s: str, i: int) -> bool:
    return s[i] == ' '


class StringView:
    def __init__(self, original_string: str, in_view: List[bool]):
        self.original_string = original_string
        self.in_view = in_view
        self.string_view = ''.join((
            ch
            for ch, is_included in zip(original_string, in_view)
            if is_included
        ))
        self.original_index_at = [
            index
            for index, is_included in enumerate(in_view)
            if is_included
        ]

    @staticmethod
    def from_view_map(original_string: str, in_view: List[bool]) -> 'StringView':
        return StringView(original_string, in_view)

    def __str__(self):
        return self.string_view

    def __repr__(self):
        return '%s(%s, %s)' % (
            type(self).__name__, self.original_string, self.in_view
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


def get_no_junk_string_view(original_string: str, isjunk: bool = NOT_SET) -> StringView:
    if isjunk == NOT_SET:
        isjunk = default_is_junk
    if isjunk is None:
        in_view = [True] * len(original_string)
    else:
        in_view = [not isjunk(original_string, index) for index, _ in enumerate(original_string)]
    return StringView.from_view_map(original_string, in_view)


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
        matching_blocks = sm.get_matching_blocks()
    else:
        matcher_is_junk_fn = space_is_junk
        haystack_string_view = get_no_junk_string_view(haystack, isjunk=matcher_is_junk_fn)
        needle_string_view = get_no_junk_string_view(needle, isjunk=matcher_is_junk_fn)
        LOGGER.debug('haystack_string_view: %s', haystack_string_view)
        LOGGER.debug('needle_string_view: %s', needle_string_view)
        sm = LocalSequenceMatcher(
            a=str(haystack_string_view),
            b=str(needle_string_view),
            scoring=DEFAULT_SCORING
        )
        matching_blocks = [
            (
                haystack_string_view.original_index_at[ai],
                needle_string_view.original_index_at[bi],
                (
                    haystack_string_view.original_index_at[ai + size - 1]
                    - haystack_string_view.original_index_at[ai]
                    + 1
                )
            )
            for ai, bi, size in sm.get_matching_blocks()
            if size
        ]
    if start_index:
        matching_blocks = [
            (ai + start_index, bi, size)
            for ai, bi, size in matching_blocks
        ]
    fm = FuzzyMatchResult(
        original_haystack,
        needle,
        matching_blocks,
        isjunk=isjunk or default_is_junk
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
