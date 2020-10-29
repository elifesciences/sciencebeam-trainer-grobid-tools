from typing import List, Tuple

from sciencebeam_trainer_grobid_tools.utils.fuzzy import (
    get_str_left_strided_matching_blocks,
    get_default_max_length_and_stride,
    fuzzy_search_index_range,
    iter_fuzzy_search_all_index_ranges
)


def _ignore_zero_length_blocks(matching_blocks: List[Tuple[int]]) -> List[Tuple[int]]:
    return [t for t in matching_blocks if t[-1]]


class TestGetStrLeftStridedMatchingBlocks:
    def test_should_return_simple_exact_match_with_large_max_length(self):
        needle = 'abc'
        haystack = needle
        assert _ignore_zero_length_blocks(get_str_left_strided_matching_blocks(
            haystack, needle,
            max_length=10, stride=5, threshold=0.8
        )) == [(0, 0, 3)]

    def test_should_return_simple_exact_match_in_larger_haystack_within_max_length(self):
        needle = 'abc'
        haystack = '0123456789' + needle
        assert _ignore_zero_length_blocks(get_str_left_strided_matching_blocks(
            haystack, needle,
            max_length=20, stride=5, threshold=0.8
        )) == [(10, 0, 3)]

    def test_should_return_simple_exact_match_in_larger_haystack_past_max_length(self):
        needle = 'abc'
        haystack = '0123456789' + needle
        assert _ignore_zero_length_blocks(get_str_left_strided_matching_blocks(
            haystack, needle,
            max_length=5, stride=5, threshold=0.8
        )) == [(10, 0, 3)]

    def test_should_return_simple_exact_match_in_larger_haystack_with_overlap(self):
        needle = 'abc'
        haystack = '0123456789' + needle
        assert _ignore_zero_length_blocks(get_str_left_strided_matching_blocks(
            haystack, needle,
            max_length=12, stride=5, threshold=0.8
        )) == [(10, 0, 3)]

    def test_should_return_skip_over_match_below_threshold(self):
        needle = 'abc'
        haystack = 'a123456789' + needle
        assert _ignore_zero_length_blocks(get_str_left_strided_matching_blocks(
            haystack, needle,
            max_length=5, stride=5, threshold=0.8
        )) == [(10, 0, 3)]


class TestGetDefaultMaxLengthAndStride:
    def test_should_calculate_max_length_and_stride(self):
        assert get_default_max_length_and_stride(
            20, 10, threshold=0.8, min_max_length=1
        ) == (48, 36)

    def test_should_use_min_max_length(self):
        assert get_default_max_length_and_stride(
            200, 10, threshold=0.8, min_max_length=100
        ) == (100, 88)

    def test_should_not_use_strides_if_haystack_less_than_min_max_length(self):
        assert get_default_max_length_and_stride(
            20, 10, threshold=0.8, min_max_length=100
        ) == (20, 20)


class TestFuzzySearchIndexRange:
    def test_should_find_exact_complete_match(self):
        haystack = 'abc'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (0, 3)

    def test_should_find_exact_match_inside(self):
        haystack = 'xyz abc 123'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (4, 7)

    def test_should_find_exact_match_surrounded_by_round_brackets(self):
        haystack = '(abc)'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_square_brackets(self):
        haystack = '[abc]'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_comma(self):
        haystack = ',abc,'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_hyphen(self):
        haystack = '-abc-'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_colon(self):
        haystack = ':abc:'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_semicolon(self):
        haystack = ';abc;'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_dot(self):
        haystack = '.abc.'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_tab(self):
        haystack = '\tabc\t'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_surrounded_by_line_feed(self):
        haystack = '\nabc\n'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_containing_dot(self):
        haystack = 'abc.'
        needle = 'abc.'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (0, 4)

    def test_should_tolerate_space_in_needle(self):
        haystack = 'abc.'
        needle = 'abc .'
        assert fuzzy_search_index_range(haystack, needle, 0.9) == (0, 4)

    def test_should_tolerate_space_in_haystack(self):
        haystack = 'abc .'
        needle = 'abc.'
        assert fuzzy_search_index_range(haystack, needle, 0.9) == (0, 5)

    def test_should_tolerate_varying_space_in_haystack_and_needle(self):
        haystack = 'Smith ,J .A .'
        needle = 'Smith, J. A.'
        assert fuzzy_search_index_range(haystack, needle, 0.5) == (0, 13)

    def test_should_find_longer_needle(self):
        haystack = 'PO Box 12345'
        needle = 'P.O. Box 12345'
        # Note: ideally it would match (0, 12)
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (3, 12)


class TestIterFuzzySearchAllIndexRanges:
    def test_should_find_exact_complete_match(self):
        haystack = 'abc'
        needle = 'abc'
        assert list(iter_fuzzy_search_all_index_ranges(haystack, needle, 0.8)) == [(0, 3)]

    def test_should_find_multiple_exact_matches(self):
        haystack = 'abc abc abc'
        needle = 'abc'
        assert list(iter_fuzzy_search_all_index_ranges(haystack, needle, 0.8)) == [
            (0, 3), (4, 7), (8, 11)
        ]
