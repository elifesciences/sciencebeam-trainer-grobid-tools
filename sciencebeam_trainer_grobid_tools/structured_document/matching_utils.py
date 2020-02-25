import logging
from itertools import islice
from typing import Callable, List, Tuple

from sciencebeam_utils.utils.collection import (
    iter_flatten
)

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
    SequenceWrapper,
    SequenceWrapperWithPosition
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    TeiText
)


LOGGER = logging.getLogger(__name__)


def iter_lines(structured_document: AbstractStructuredDocument):
    return (
        line
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
    )


def get_untagged_line_tokens(structured_document: AbstractStructuredDocument, line):
    return [
        token
        for token in structured_document.get_tokens_of_line(line)
        if not structured_document.get_tag(token)
    ]


def join_tokens_text(tokens: List[TeiText]) -> str:
    return ' '.join([token.text for token in tokens])


class PendingSequences:
    def __init__(self, sequences: List[SequenceWrapper]):
        self._sequences = sequences

    def iter_pending_sequences(self, limit: int = None):
        untagged_pending_sequences = iter_flatten(
            seq.untagged_sub_sequences() for seq in self._sequences
        )
        if limit:
            untagged_pending_sequences = islice(untagged_pending_sequences, limit)
        return untagged_pending_sequences

    def get_pending_sequences(self, limit: int = None):
        return list(self.iter_pending_sequences(limit=limit))

    @staticmethod
    def from_structured_document(
            structured_document: AbstractStructuredDocument, normalize_fn: Callable[[str], str]):
        pending_sequences = []
        for line in iter_lines(structured_document):
            tokens = get_untagged_line_tokens(structured_document, line)
            if tokens:
                LOGGER.debug(
                    'tokens without tag: %s',
                    [structured_document.get_text(token) for token in tokens]
                )
                pending_sequences.append(SequenceWrapperWithPosition(
                    structured_document,
                    tokens,
                    normalize_fn,
                    position=len(pending_sequences)
                ))
        return PendingSequences(pending_sequences)


def _join_with_index_ranges(
        items: List[str], sep: str, pad: str = '') -> Tuple[str, List[Tuple[int, int]]]:
    item_str_list = [pad + str(item) + pad for item in items]
    text = sep.join(item_str_list)
    item_start = len(pad)
    item_sep_pad_len = len(sep) + 2 * len(pad)
    index_ranges = []
    for item_str in item_str_list:
        item_end = item_start + len(item_str)
        index_ranges.append((item_start, item_end))
        item_start = item_end + item_sep_pad_len
    return text, index_ranges


class JoinedText:
    def __init__(self, items: List[str], sep: str, pad: str = ''):
        self._items = items
        self._text, self._item_index_ranges = _join_with_index_ranges(items, sep=sep, pad=pad)

    def iter_items_and_index_range_between(self, index_range: Tuple[int, int]):
        start, end = index_range
        for item, (item_start, item_end) in zip(self._items, self._item_index_ranges):
            if item_start >= end:
                break
            if item_end > start:
                yield item, (item_start, item_end)

    def get_text(self):
        return self._text

    def __str__(self):
        return self.get_text()


class SequencesText:
    def __init__(self, sequences: List[SequenceWrapper], sep: str = '\n', pad: str = ''):
        self._joined_text = JoinedText(sequences, sep=sep, pad=pad)

    def iter_tokens_between(self, index_range: Tuple[int, int]):
        start, end = index_range
        seq_and_index_range_iterable = self._joined_text.iter_items_and_index_range_between(
            index_range
        )
        for seq, (seq_start, _) in seq_and_index_range_iterable:
            yield from seq.tokens_between((start - seq_start, end - seq_start))

    def get_text_between(self, index_range: Tuple[int, int]):
        return join_tokens_text(self.iter_tokens_between(index_range))

    def get_index_ranges_with_text(
            self, index_ranges: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], str]]:
        return list(zip(
            index_ranges,
            map(self.get_text_between, index_ranges)
        ))

    def get_text(self):
        return self._joined_text.get_text()

    def __str__(self):
        return self.get_text()
