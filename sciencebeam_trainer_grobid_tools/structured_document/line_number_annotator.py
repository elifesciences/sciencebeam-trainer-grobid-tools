import logging

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)


LOGGER = logging.getLogger(__name__)


# Among first tokens on each line, minimum ratio of line number vs other non-numeric tokens
# (low ratio indicates numbers may be figures or table values rather than line numbers)
DEFAULT_LINE_NUMBER_RATIO_THRESHOLD = 0.7


DEFAULT_LINE_NO_TAG = 'line_no'


def iter_first_tokens_of_lines(
        structured_document: GrobidTrainingTeiStructuredDocument,
        lines: list):
    for line in lines:
        text_tokens = structured_document.get_tokens_of_line(line)
        if text_tokens:
            yield text_tokens[0]


def get_line_number_candidates(
        structured_document: GrobidTrainingTeiStructuredDocument,
        tokens: list):
    line_number_candidates_with_num = [
        (token, int(structured_document.get_text(token)))
        for token in tokens
        if structured_document.get_text(token).isdigit()
    ]
    if not line_number_candidates_with_num:
        return []
    line_number_candidates_with_num = sorted(
        line_number_candidates_with_num,
        key=lambda pair: pair[1]
    )
    line_number_sequences = [[line_number_candidates_with_num[0]]]
    for token, num in line_number_candidates_with_num[1:]:
        prev_seq = line_number_sequences[-1]
        prev_num = prev_seq[-1][1]
        if num == prev_num + 1:
            prev_seq.append((token, num))
        else:
            line_number_sequences.append([(token, num)])
    max_line_number_sequence = max(map(len, line_number_sequences))
    LOGGER.debug(
        'line_number_sequences (max len: %d): %s',
        max_line_number_sequence, line_number_sequences
    )
    longest_line_number_sequence = [
        seq
        for seq in line_number_sequences
        if len(seq) == max_line_number_sequence
    ][0]
    LOGGER.debug(
        'longest_line_number_sequence (len: %d): %s',
        len(longest_line_number_sequence), longest_line_number_sequence
    )
    return [token for token, _ in longest_line_number_sequence]


def iter_find_line_number_tokens_in_lines(
        structured_document: GrobidTrainingTeiStructuredDocument,
        lines: list,
        min_line_number: int,
        line_number_ratio_threshold: float):
    first_tokens_of_lines = list(iter_first_tokens_of_lines(
        structured_document,
        lines
    ))
    line_number_candidates = get_line_number_candidates(
        structured_document,
        first_tokens_of_lines
    )
    if len(line_number_candidates) < min_line_number:
        LOGGER.debug('not enough line number candidates: %d', len(line_number_candidates))
        return []
    liner_number_ratio = len(line_number_candidates) / len(first_tokens_of_lines)
    if liner_number_ratio < line_number_ratio_threshold:
        LOGGER.debug(
            'line number ratio not met: %.3f < %.3f',
            liner_number_ratio, line_number_ratio_threshold
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
            min_line_number: int = 2,
            line_number_ratio_threshold: float = DEFAULT_LINE_NUMBER_RATIO_THRESHOLD):
        self.tag = tag
        self.min_line_number = min_line_number
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
            line_number_ratio_threshold=self.config.line_number_ratio_threshold
        )
        for t in line_number_tokens:
            structured_document.set_tag(t, self.config.tag)
        return structured_document
