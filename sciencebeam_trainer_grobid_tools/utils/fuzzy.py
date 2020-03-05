import logging
from typing import Tuple

from sciencebeam_alignment.align import (
    LocalSequenceMatcher
)

from sciencebeam_alignment.word_sequence_matcher import (
    WordSequenceMatcher
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    FuzzyMatchResult,
    DEFAULT_SCORING
)


LOGGER = logging.getLogger(__name__)


def fuzzy_search(
        haystack: str, needle: str,
        threshold: float,
        exact_word_match_threshold: int = 5) -> FuzzyMatchResult:
    if min(len(haystack), len(needle)) < exact_word_match_threshold:
        sm = WordSequenceMatcher(None, haystack, needle)
    else:
        sm = LocalSequenceMatcher(a=haystack, b=needle, scoring=DEFAULT_SCORING)
    matching_blocks = sm.get_matching_blocks()
    fm = FuzzyMatchResult(haystack, needle, matching_blocks)
    LOGGER.debug('fm (sm=%s, threshold=%.2f): %s', type(sm).__name__, threshold, fm)
    if fm.b_gap_ratio() >= threshold:
        return fm
    return None


def fuzzy_search_index_range(*args, **kwargs) -> Tuple[int, int]:
    fm = fuzzy_search(*args, **kwargs)
    if fm:
        return fm.a_index_range()
    return None
