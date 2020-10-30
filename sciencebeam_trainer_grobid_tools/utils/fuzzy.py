import logging
from difflib import SequenceMatcher
from typing import List, Tuple

from sciencebeam_alignment.align import (
    LocalSequenceMatcher
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    FuzzyMatchResult as _FuzzyMatchResult,
    len_index_range,
    DEFAULT_SCORING
)


LOGGER = logging.getLogger(__name__)


DEFAULT_WORD_SEPARATORS = ' .,-:;()[]\n\t'

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
    return s[i] in {' ', '\t', '\n'}


class FuzzyMatchResult(_FuzzyMatchResult):
    # Note: fixing calculation
    def b_gap_ratio(self):
        """
        Calculates the ratio of matches vs the length of b,
        but also adds any gaps / mismatches within a.
        """
        a_index_range = self.a_index_range()
        a_match_len = len_index_range(a_index_range)
        match_count = self.match_count()
        a_junk_match_count = self.a_non_matching_junk_count(a_index_range)
        b_junk_count = self.b_non_matching_junk_count()
        a_gaps = max(0, a_match_len - match_count)
        # LOGGER.debug(
        #     'len b: %d, a gaps: %d, a junk: %d, b junk: %d, a_index_range: %s',
        #     len(self.b), a_gaps, a_junk_match_count, b_junk_count, a_index_range
        # )
        return self.ratio_to(len(self.b) + a_gaps - a_junk_match_count - b_junk_count)


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


def get_str_matching_blocks_using_sequence_matcher(
        haystack: str, needle: str) -> List[Tuple[int, int, int]]:
    sm = LocalSequenceMatcher(a=haystack, b=needle, scoring=DEFAULT_SCORING)
    return sm.get_matching_blocks()


# def get_str_matching_blocks_using_fuzzy_search(
#         haystack: str, needle: str, threshold: float) -> List[Tuple[int, int, int]]:
#     max_l_dist = round(min(len(haystack), len(needle)) * (1 - threshold))
#     matches = find_near_matches(needle, haystack, max_l_dist=max_l_dist)
#     if not matches:
#         return []
#     # pretend there is just one matching block
#     match = matches[0]
#     return [(
#         match.start,
#         0,
#         match.end - match.start
#     )]


# def get_str_matching_blocks(
#         haystack: str, needle: str) -> List[Tuple[int, int, int]]:
#     return get_str_matching_blocks_using_sequence_matcher(
#         haystack, needle
#     )


def get_matching_blocks_b_gap_ratio(
        haystack: str,
        needle: str,
        matching_blocks,
        isjunk: callable) -> float:
    fm = FuzzyMatchResult(
        haystack,
        needle,
        matching_blocks,
        isjunk=isjunk
    )
    LOGGER.debug('temp fm: %s', fm)
    return fm.b_gap_ratio()


def get_matching_blocks_size(matching_blocks: List[Tuple[int]]) -> int:
    return sum(size for _, _, size in matching_blocks)


def get_first_chunk_matching_blocks(
        haystack: str,
        needle: str,
        matching_blocks,
        threshold: float,
        isjunk: callable) -> float:
    block_count = len(matching_blocks) - 1
    while block_count:
        chunk_matching_blocks = matching_blocks[:block_count]
        chunk_match_count = get_matching_blocks_size(chunk_matching_blocks)
        chunk_needle = needle[:chunk_match_count]
        fm = FuzzyMatchResult(
            haystack,
            chunk_needle,
            chunk_matching_blocks,
            isjunk=isjunk
        )
        LOGGER.debug('temp fm: %s', fm)
        if fm.b_gap_ratio() >= threshold:
            return chunk_matching_blocks
    return []


def get_str_left_strided_matching_blocks(
        haystack: str, needle: str,
        max_length: int,
        stride: int,
        threshold: float,
        isjunk: callable = None,
        max_chunks: int = 1,
        start_index: int = 0) -> List[Tuple[int, int, int]]:
    """
    LocalSequenceMatcher scales quadratically (O(n*m) with n=haystack length and m=needle length)
    By using a window, we are limiting the memory usage and stop early if we found a match.
    """
    max_offset = stride
    while start_index < len(haystack):
        LOGGER.debug('start_index: %d, threshold: %.3f', start_index, threshold)
        matching_blocks = get_str_matching_blocks_using_sequence_matcher(
            haystack[start_index:(start_index + max_length)],
            needle
        )
        if (
            not matching_blocks
            or matching_blocks[0][0] > max_offset
            or not matching_blocks[0][2]
        ):
            start_index += stride
            continue
        if (
            get_matching_blocks_b_gap_ratio(
                haystack, needle, matching_blocks, isjunk=isjunk
            ) < threshold
        ):
            if max_chunks <= 1:
                start_index += stride
                continue
            first_chunk_matching_blocks = get_first_chunk_matching_blocks(
                haystack=haystack,
                needle=needle,
                matching_blocks=matching_blocks,
                threshold=threshold,
                isjunk=isjunk
            )
            first_chunk_match_count = get_matching_blocks_size(first_chunk_matching_blocks)
            if not first_chunk_match_count:
                start_index += stride
                continue
            remaining_needle = needle[first_chunk_match_count:]
            remaining_matching_blocks = get_str_left_strided_matching_blocks(
                haystack=haystack,
                needle=remaining_needle,
                max_length=max_length,
                stride=stride,
                threshold=threshold,
                isjunk=isjunk,
                max_chunks=max_chunks - 1,
                start_index=start_index + first_chunk_match_count
            )
            if not remaining_matching_blocks:
                start_index += stride
                continue
            return first_chunk_matching_blocks + [
                (ai, bi + first_chunk_match_count, size)
                for ai, bi, size in remaining_matching_blocks
            ]
        if not start_index:
            return matching_blocks
        return [
            (ai + start_index, bi, size)
            for ai, bi, size in matching_blocks
        ]
    return []


def get_default_max_length_and_stride(
        haystack_length: int,
        needle_length: int,
        threshold: float,
        min_max_length: int = 1000) -> Tuple[int]:
    if haystack_length <= min_max_length:
        return haystack_length, haystack_length
    max_l_dist = round(min(haystack_length, needle_length) * (1 - threshold))
    max_matched_needle_length = needle_length + max_l_dist
    max_length = max(min_max_length, max_matched_needle_length * 4)
    stride = max_length - max_matched_needle_length
    return max_length, stride


def get_str_auto_left_strided_matching_blocks(
        haystack: str, needle: str,
        threshold: float,
        isjunk: callable) -> List[Tuple[int, int, int]]:
    max_length, stride = get_default_max_length_and_stride(
        len(haystack), len(needle), threshold=threshold
    )
    return get_str_left_strided_matching_blocks(
        haystack, needle,
        max_length=max_length, stride=stride, threshold=threshold, isjunk=isjunk
    )


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
        mode = 'word'
        sm = WordSequenceMatcher(None, haystack, needle, sep=DEFAULT_WORD_SEPARATORS)
        matching_blocks = sm.get_matching_blocks()
    else:
        mode = 'char'
        matcher_is_junk_fn = space_is_junk
        haystack_string_view = get_no_junk_string_view(haystack, isjunk=matcher_is_junk_fn)
        needle_string_view = get_no_junk_string_view(needle, isjunk=matcher_is_junk_fn)
        LOGGER.debug('haystack_string_view: %r', str(haystack_string_view))
        LOGGER.debug('needle_string_view: %r', str(needle_string_view))
        raw_matching_blocks = get_str_auto_left_strided_matching_blocks(
            haystack=str(haystack_string_view),
            needle=str(needle_string_view),
            threshold=threshold,
            isjunk=isjunk or default_is_junk
        )
        LOGGER.debug('raw_matching_blocks: %s', raw_matching_blocks)
        # if str(needle_string_view) == 'pmc1000001':
        #     raise RuntimeError('dummy')
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
            for ai, bi, size in raw_matching_blocks
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
    LOGGER.debug('fm (mode=%s, threshold=%.2f): %s', mode, threshold, fm)
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
