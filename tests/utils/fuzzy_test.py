from sciencebeam_trainer_grobid_tools.utils.fuzzy import (
    fuzzy_search_index_range,
    iter_fuzzy_search_all_index_ranges
)


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

    def test_should_find_exact_match_surrounded_by_dot(self):
        haystack = '.abc.'
        needle = 'abc'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (1, 4)

    def test_should_find_exact_match_containing_dot(self):
        haystack = 'abc.'
        needle = 'abc.'
        assert fuzzy_search_index_range(haystack, needle, 0.8) == (0, 4)


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
