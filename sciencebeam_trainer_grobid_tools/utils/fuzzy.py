import logging
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Tuple

from sciencebeam_alignment.align import (
    LocalSequenceMatcher
)

from sciencebeam_trainer_grobid_tools.core.annotation.fuzzy_match import (
    FuzzyMatchResult as _FuzzyMatchResult,
    len_index_range,
    DEFAULT_SCORING
)


LOGGER = logging.getLogger(__name__)


DEFAULT_WORD_SEPARATORS = ' .,-:;()[]\n\t'

NOT_SET = 'NOT_SET'


def default_is_junk(s, i):
    try:
        ch = s[i]
    except IndexError as exc:
        raise IndexError('index out of range: %s (len: %d, s: %r)' % (i, len(s), s)) from exc
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


def get_merged_matching_blocks_chunks(
        matching_blocks_chunks: List[List[Tuple[int]]]) -> List[Tuple[int, int, int]]:
    return [
        matching_block
        for matching_blocks in matching_blocks_chunks
        for matching_block in matching_blocks
    ]


class ChunkedFuzzyMatchResult:
    def __init__(self, matches: List[FuzzyMatchResult]):
        self.matches = matches

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.matches)

    def merge(self):
        return FuzzyMatchResult(
            self.matches[0].a,
            self.matches[0].b,
            get_merged_matching_blocks_chunks([
                fm.matching_blocks
                for fm in self.matches
            ]),
            isjunk=self.matches[0].isjunk
        )


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


def get_matching_blocks_end_offset(matching_blocks: List[Tuple[int]], seq_index: int) -> int:
    if not matching_blocks:
        return 0
    last_block = matching_blocks[-1]
    last_block_size = last_block[2]
    if not last_block_size:
        return 0
    return last_block[seq_index] + last_block_size


def get_matching_blocks_start_offset(matching_blocks: List[Tuple[int]], seq_index: int) -> int:
    if not matching_blocks:
        return None
    first_block = matching_blocks[0]
    first_block_size = first_block[2]
    if not first_block_size:
        return None
    return first_block[seq_index]


def get_first_chunk_matching_blocks(
        haystack: str,
        needle: str,
        matching_blocks,
        threshold: float,
        isjunk: callable) -> List[Tuple[int]]:
    block_count = len(matching_blocks) - 1
    while block_count:
        chunk_matching_blocks = matching_blocks[:block_count]
        chunk_needle_end = get_matching_blocks_end_offset(chunk_matching_blocks, 1)
        LOGGER.debug('chunk_needle_end: %s', chunk_needle_end)
        if not chunk_needle_end:
            break
        chunk_needle = needle[:chunk_needle_end]
        fm = FuzzyMatchResult(
            haystack,
            chunk_needle,
            chunk_matching_blocks,
            isjunk=isjunk
        )
        LOGGER.debug('temp fm: %s', fm)
        if fm.b_gap_ratio() >= threshold:
            LOGGER.debug('chunk_needle: %s', chunk_needle)
            return chunk_matching_blocks
        block_count -= 1
    return []


def get_last_chunk_matching_blocks(
        haystack: str,
        needle: str,
        matching_blocks,
        threshold: float,
        isjunk: callable) -> List[Tuple[int]]:
    block_start = 0
    while block_start < len(matching_blocks):
        chunk_matching_blocks = matching_blocks[block_start:]
        chunk_needle_start = get_matching_blocks_start_offset(chunk_matching_blocks, 1)
        LOGGER.debug('chunk_needle_start: %s', chunk_needle_start)
        if chunk_needle_start is None:
            break
        chunk_needle = needle[chunk_needle_start:]
        offset_chunk_matching_blocks = get_offset_matching_blocks(
            chunk_matching_blocks,
            b_offset=(0 - chunk_needle_start)
        )
        fm = FuzzyMatchResult(
            haystack,
            chunk_needle,
            offset_chunk_matching_blocks,
            isjunk=isjunk
        )
        LOGGER.debug('temp fm: %s', fm)
        if fm.b_gap_ratio() >= threshold:
            LOGGER.debug('chunk_needle: %s', chunk_needle)
            return chunk_matching_blocks
        block_start += 1
    return []


def get_first_or_last_chunk_matching_blocks(*args, **kwargs) -> List[Tuple[int]]:
    first_chunk_matching_blocks = get_first_chunk_matching_blocks(*args, **kwargs)
    if first_chunk_matching_blocks:
        return first_chunk_matching_blocks, None
    return None, get_last_chunk_matching_blocks(*args, **kwargs)


def get_offset_matching_blocks(
        matching_blocks: List[Tuple[int]],
        a_offset: int = 0,
        b_offset: int = 0) -> List[Tuple[int]]:
    if not a_offset and not b_offset:
        return matching_blocks
    return [
        (ai + a_offset, bi + b_offset, size)
        for ai, bi, size in matching_blocks
    ]


def get_str_left_strided_matching_blocks_chunks(
        haystack: str, needle: str,
        max_length: int,
        stride: int,
        threshold: float,
        isjunk: callable = None,
        max_chunks: int = 1,
        start_index: int = 0) -> List[List[Tuple[int, int, int]]]:
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
            (
                first_chunk_matching_blocks,
                last_chunk_matching_blocks
            ) = get_first_or_last_chunk_matching_blocks(
                haystack=haystack,
                needle=needle,
                matching_blocks=matching_blocks,
                threshold=threshold,
                isjunk=isjunk
            )
            LOGGER.debug(
                'first_chunk_matching_blocks: %s, last_chunk_matching_blocks: %s',
                first_chunk_matching_blocks, last_chunk_matching_blocks
            )
            if not first_chunk_matching_blocks and not last_chunk_matching_blocks:
                start_index += stride
                continue
            if first_chunk_matching_blocks:
                first_chunk_needle_start = 0
                first_chunk_needle_end = get_matching_blocks_end_offset(
                    first_chunk_matching_blocks, 1
                )
                remaining_needle = needle[first_chunk_needle_end:]
                remaining_start_index = start_index + first_chunk_needle_end
            else:
                first_chunk_needle_start = get_matching_blocks_start_offset(
                    last_chunk_matching_blocks, 1
                )
                first_chunk_needle_end = len(needle)
                remaining_needle = needle[:first_chunk_needle_start]
                remaining_start_index = 0
            remaining_matching_blocks_chunks = get_str_left_strided_matching_blocks_chunks(
                haystack=haystack,
                needle=remaining_needle,
                max_length=max_length,
                stride=stride,
                threshold=threshold,
                isjunk=isjunk,
                max_chunks=max_chunks - 1,
                start_index=remaining_start_index
            )
            LOGGER.debug('remaining_matching_blocks_chunks: %s', remaining_matching_blocks_chunks)
            if not remaining_matching_blocks_chunks:
                start_index += stride
                continue
            if last_chunk_matching_blocks:
                return remaining_matching_blocks_chunks + [last_chunk_matching_blocks]
            return [first_chunk_matching_blocks] + [
                get_offset_matching_blocks(
                    remaining_matching_blocks,
                    b_offset=first_chunk_needle_end
                )
                for remaining_matching_blocks in remaining_matching_blocks_chunks
            ]
        if not start_index:
            return [matching_blocks]
        return [[
            (ai + start_index, bi, size)
            for ai, bi, size in matching_blocks
        ]]
    return []


def get_str_left_strided_matching_blocks(*args, **kwargs) -> List[Tuple[int, int, int]]:
    matching_blocks_chunks = get_str_left_strided_matching_blocks_chunks(*args, **kwargs)
    return get_merged_matching_blocks_chunks(matching_blocks_chunks)


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


def get_str_auto_left_strided_matching_blocks_chunks(
        haystack: str, needle: str,
        threshold: float,
        **kwargs) -> List[List[Tuple[int, int, int]]]:
    max_length, stride = get_default_max_length_and_stride(
        len(haystack), len(needle), threshold=threshold
    )
    return get_str_left_strided_matching_blocks_chunks(
        haystack, needle,
        max_length=max_length, stride=stride, threshold=threshold,
        **kwargs
    )


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


def fuzzy_search_chunks(
        haystack: str, needle: str,
        threshold: float,
        exact_word_match_threshold: int = 5,
        max_chunks: int = 1,
        start_index: int = 0,
        isjunk: callable = None) -> ChunkedFuzzyMatchResult:
    original_haystack = haystack
    if start_index:
        haystack = haystack[start_index:]
    if min(len(haystack), len(needle)) < exact_word_match_threshold:
        mode = 'word'
        sm = WordSequenceMatcher(None, haystack, needle, sep=DEFAULT_WORD_SEPARATORS)
        matching_blocks = sm.get_matching_blocks()
        matching_blocks = get_offset_matching_blocks(matching_blocks, a_offset=start_index)
        fm = FuzzyMatchResult(
            original_haystack,
            needle,
            matching_blocks,
            isjunk=isjunk or default_is_junk
        )
        LOGGER.debug('fm (mode=%s, threshold=%.2f): %s', mode, threshold, fm)
        if fm.b_gap_ratio() < threshold:
            return None
        return ChunkedFuzzyMatchResult([fm])
    mode = 'char'
    matcher_is_junk_fn = space_is_junk
    haystack_string_view = get_no_junk_string_view(haystack, isjunk=matcher_is_junk_fn)
    needle_string_view = get_no_junk_string_view(needle, isjunk=matcher_is_junk_fn)
    LOGGER.debug('haystack_string_view: %r', str(haystack_string_view))
    LOGGER.debug('needle_string_view: %r', str(needle_string_view))
    raw_matching_blocks_chunks = get_str_auto_left_strided_matching_blocks_chunks(
        haystack=str(haystack_string_view),
        needle=str(needle_string_view),
        threshold=threshold,
        max_chunks=max_chunks,
        isjunk=isjunk or default_is_junk
    )
    LOGGER.debug('raw_matching_blocks_chunks: %s', raw_matching_blocks_chunks)
    if not raw_matching_blocks_chunks:
        return None
    matching_blocks_chunks = [
        [
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
        for raw_matching_blocks in raw_matching_blocks_chunks
    ]
    matching_blocks_chunks = [
        get_offset_matching_blocks(
            matching_blocks,
            a_offset=start_index
        )
        for matching_blocks in matching_blocks_chunks
    ]
    fm_chunks = ChunkedFuzzyMatchResult([
        FuzzyMatchResult(
            original_haystack,
            needle,
            matching_blocks,
            isjunk=isjunk or default_is_junk
        )
        for matching_blocks in matching_blocks_chunks
    ])
    LOGGER.debug('fm_chunks (mode=%s, threshold=%.2f): %s', mode, threshold, fm_chunks)
    return fm_chunks


def fuzzy_search(*args, **kwargs) -> FuzzyMatchResult:
    fms = fuzzy_search_chunks(*args, **kwargs)
    if not fms:
        return None
    return fms.merge()


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


def fuzzy_search_index_range(*args, **kwargs) -> Optional[Tuple[int, int]]:
    fm = fuzzy_search(*args, **kwargs)
    if fm:
        return fm.a_index_range()
    return None


def fuzzy_search_index_range_chunks(*args, **kwargs) -> Optional[List[Tuple[int, int]]]:
    fm_chunks = fuzzy_search_chunks(*args, **kwargs)
    if not fm_chunks:
        return None
    return [
        fm.a_index_range()
        for fm in fm_chunks.matches
    ]


def iter_fuzzy_search_all_index_ranges(*args, **kwargs) -> Iterable[Tuple[int, int]]:
    return (
        fm.a_index_range()
        for fm in iter_fuzzy_search_all(*args, **kwargs)
    )
