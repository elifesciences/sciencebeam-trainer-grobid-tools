import logging

from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)


LOGGER = logging.getLogger(__name__)


DEFAULT_MIN_LINE_NUMBER_COUNT = 10

DEFAULT_MAX_LINE_NUMBER_GAP = 10


# Among first tokens on each line, minimum ratio of line number vs other non-numeric tokens
# (low ratio indicates numbers may be figures or table values rather than line numbers)
DEFAULT_LINE_NUMBER_RATIO_THRESHOLD = 0.1


DEFAULT_LINE_NO_TAG = 'line_no'


def iter_first_tokens_of_lines(
        structured_document: GrobidTrainingTeiStructuredDocument,
        lines: list):
    for line in lines:
        text_tokens = structured_document.get_tokens_of_line(line)
        if text_tokens:
            yield text_tokens[0]


def parse_line_number(text: str) -> int:
    return int(text)


def is_line_number(text: str) -> int:
    try:
        value = parse_line_number(text)
        return value > 0
    except ValueError:
        return False


def get_line_number_candidates(
        structured_document: GrobidTrainingTeiStructuredDocument,
        tokens: list,
        min_line_number: int,
        max_line_number_gap: int):
    line_number_candidates_with_num = [
        (token, parse_line_number(structured_document.get_text(token)), 1 + index)
        for index, token in enumerate(tokens)
        if is_line_number(structured_document.get_text(token))
    ]
    if not line_number_candidates_with_num:
        return []
    line_number_candidates_with_num = sorted(
        line_number_candidates_with_num,
        key=lambda item: (item[1], item[2])
    )
    line_number_sequences = [[line_number_candidates_with_num[0]]]
    for item in line_number_candidates_with_num[1:]:
        token, num, token_pos = item
        prev_seq = line_number_sequences[-1]
        prev_item = prev_seq[-1]
        _, prev_num, prev_token_pos = prev_item
        expected_num = prev_num + 1
        if token_pos < prev_token_pos or num == prev_num:
            LOGGER.debug('ignoring out of sequence: %s (prev: %s)', item, prev_item)
        elif expected_num <= num <= expected_num + max_line_number_gap:
            prev_seq.append(item)
        else:
            line_number_sequences.append([item])
    accepted_line_number_sequences = [
        seq
        for seq in line_number_sequences
        if len(seq) >= min_line_number
    ]
    return [
        token
        for seq in accepted_line_number_sequences
        for token, _, _ in seq
    ]


def iter_find_line_number_tokens_in_lines(
        structured_document: GrobidTrainingTeiStructuredDocument,
        lines: list,
        min_line_number: int,
        max_line_number_gap: int,
        line_number_ratio_threshold: float):
    first_tokens_of_lines = list(iter_first_tokens_of_lines(
        structured_document,
        lines
    ))
    line_number_candidates = get_line_number_candidates(
        structured_document,
        first_tokens_of_lines,
        min_line_number=min_line_number,
        max_line_number_gap=max_line_number_gap
    )
    if len(line_number_candidates) < min_line_number:
        LOGGER.debug('not enough line number candidates: %d', len(line_number_candidates))
        return []
    line_number_ratio = len(line_number_candidates) / len(first_tokens_of_lines)
    if line_number_ratio < line_number_ratio_threshold:
        LOGGER.debug('first_tokens_of_lines: %s', first_tokens_of_lines)
        LOGGER.debug(
            'line number ratio not met: %.3f < %.3f',
            line_number_ratio, line_number_ratio_threshold
        )
        return []
    return line_number_candidates


def iter_find_line_number_tokens(
        structured_document: GrobidTrainingTeiStructuredDocument,
        **kwargs):

    for page in structured_document.get_pages():
        line_no_tokens = iter_find_line_number_tokens_in_lines(
            structured_document,
            lines=structured_document.get_lines_of_page(page),
            **kwargs
        )
        yield from line_no_tokens


class TextLineNumberAnnotatorConfig:
    def __init__(
            self,
            tag: str = DEFAULT_LINE_NO_TAG,
            min_line_number: int = DEFAULT_MIN_LINE_NUMBER_COUNT,
            max_line_number_gap: int = DEFAULT_MAX_LINE_NUMBER_GAP,
            line_number_ratio_threshold: float = DEFAULT_LINE_NUMBER_RATIO_THRESHOLD):
        self.tag = tag
        self.min_line_number = min_line_number
        self.max_line_number_gap = max_line_number_gap
        self.line_number_ratio_threshold = line_number_ratio_threshold


# Similar to: sciencebeam_gym.preprocess.annotation.annotator.LineAnnotator
# But this implementation does not require coordinates
class TextLineNumberAnnotator(AbstractAnnotator):
    def __init__(
            self,
            config: TextLineNumberAnnotatorConfig = None):
        if config is None:
            config = TextLineNumberAnnotatorConfig()
        self.config = config

    def annotate(
            self,
            structured_document: GrobidTrainingTeiStructuredDocument
            ) -> GrobidTrainingTeiStructuredDocument:
        line_number_tokens = iter_find_line_number_tokens(
            structured_document,
            min_line_number=self.config.min_line_number,
            max_line_number_gap=self.config.max_line_number_gap,
            line_number_ratio_threshold=self.config.line_number_ratio_threshold
        )
        for t in line_number_tokens:
            structured_document.set_tag(t, self.config.tag)
        return structured_document
