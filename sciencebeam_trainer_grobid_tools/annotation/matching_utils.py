import logging
from itertools import islice
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast
)

from sciencebeam_utils.utils.collection import (
    iter_flatten
)

from sciencebeam_trainer_grobid_tools.core.annotation.fuzzy_match import remove_junk

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    TeiText
)


LOGGER = logging.getLogger(__name__)


THIN_SPACE = u'\u2009'
EN_DASH = u'\u2013'
EM_DASH = u'\u2014'
APOS = u'&apos;'

T_Item = TypeVar('T_Item')
T_Token = TypeVar('T_Token')


def DEFAULT_ISJUNK(*_, **__):
    return False


def normalise_str(s):
    return (
        s.lower().replace(EM_DASH, u'-').replace(EN_DASH, u'-').replace(THIN_SPACE, ' ')
        .replace('&apos;', '"')
        .replace('\'', '"')
    )


def normalise_str_or_list(x):
    if isinstance(x, list):
        return [normalise_str(s) for s in x]
    else:
        return normalise_str(x)


def normalise_and_remove_junk_str(
        text: str,
        is_junk_fn: Callable[[str, int], bool] = None) -> str:
    if is_junk_fn is None:
        is_junk_fn = DEFAULT_ISJUNK
    return remove_junk(normalise_str(text), is_junk_fn)


T_Text_Or_Text_List = TypeVar('T_Text_Or_Text_List', bound=Union[str, List[str]])


def normalise_and_remove_junk_str_or_list(
    text_or_list: T_Text_Or_Text_List,
    is_junk_fn: Callable[[str, int], bool] = None
) -> T_Text_Or_Text_List:
    if isinstance(text_or_list, list):
        return cast(T_Text_Or_Text_List, [
            normalise_and_remove_junk_str(s, is_junk_fn=is_junk_fn)
            for s in text_or_list
        ])
    else:
        return cast(T_Text_Or_Text_List, normalise_and_remove_junk_str(
            cast(str, text_or_list),
            is_junk_fn=is_junk_fn
        ))


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


def get_token_whitespace(token: T_Token) -> str:
    whitespace = getattr(token, 'whitespace', None)
    if whitespace is not None:
        return whitespace
    return ' '


def join_with_index_ranges(
    items: List[T_Item],
    sep: str,
    pad: str = '',
    whitespace_list: List[str] = None
) -> Tuple[str, List[Tuple[int, int]]]:
    item_str_list = [pad + str(item) + pad for item in items]
    if whitespace_list:
        whitespace_list = [
            whitespace if whitespace is not None else sep
            for index, whitespace in enumerate(whitespace_list)
        ]
        whitespace_list[-1] = ''
        text = ''.join(iter_flatten(zip(item_str_list, whitespace_list)))
    else:
        text = sep.join(item_str_list)
    item_start = len(pad)
    item_sep_pad_len = len(sep) + 2 * len(pad)
    index_ranges = []
    for index, item_str in enumerate(item_str_list):
        item_end = item_start + len(item_str)
        index_ranges.append((item_start, item_end))
        if whitespace_list:
            item_start = item_end + len(whitespace_list[index]) + 2 * len(pad)
        else:
            item_start = item_end + item_sep_pad_len
    return text, index_ranges


class JoinedText(Generic[T_Item]):
    def __init__(
        self,
        items: List[T_Item],
        sep: str,
        pad: str = '',
        whitespace_list: List[str] = None
    ):
        self._items = items
        self._text, self._item_index_ranges = join_with_index_ranges(
            items, sep=sep, pad=pad, whitespace_list=whitespace_list
        )

    @property
    def end_index(self):
        if not self._item_index_ranges:
            return 0
        last_index_range = self._item_index_ranges[-1]
        _, last_index_end = last_index_range
        return last_index_end

    def iter_item_indices_and_index_range_between(self, index_range: Tuple[int, int]):
        start, end = index_range
        for item_index, (item_start, item_end) in enumerate(self._item_index_ranges):
            if item_start >= end:
                break
            if item_end > start:
                yield item_index, (item_start, item_end)

    def iter_items_and_index_range_between(self, index_range: Tuple[int, int]):
        return (
            (self._items[index], item_index_range)
            for index, item_index_range in self.iter_item_indices_and_index_range_between(
                index_range
            )
        )

    def get_text(self):
        return self._text

    def __str__(self):
        return self.get_text()


class SequenceWrapper(Generic[T_Token]):
    def __init__(
            self,
            structured_document: AbstractStructuredDocument,
            tokens: List[T_Token],
            str_filter_f: Optional[Callable[[str], str]] = None):
        self.structured_document = structured_document
        self.str_filter_f = str_filter_f
        self.tokens = tokens
        self.token_str_list = [structured_document.get_text(t) or '' for t in tokens]
        if str_filter_f:
            self.token_str_list = [str_filter_f(s) for s in self.token_str_list]
        self.joined_text = JoinedText(
            self.token_str_list,
            sep=' ',
            whitespace_list=[
                get_token_whitespace(t) for t in tokens
            ]
        )
        self.tokens_as_str = str(self.joined_text)

    def tokens_between(self, index_range: Tuple[int, int]) -> Iterable[T_Token]:
        for index, _ in self.joined_text.iter_item_indices_and_index_range_between(index_range):
            yield self.tokens[index]

    def sub_sequence_for_tokens(self, tokens: List[T_Token]):
        return SequenceWrapper(self.structured_document, tokens, str_filter_f=self.str_filter_f)

    def untagged_sub_sequences(self) -> Iterable['SequenceWrapper']:
        token_tags = [self.structured_document.get_tag(t) for t in self.tokens]
        tagged_count = len([t for t in token_tags if t])
        if tagged_count == 0:
            yield self
        elif tagged_count == len(self.tokens):
            pass
        else:
            untagged_tokens = []
            for token, tag in zip(self.tokens, token_tags):
                if not tag:
                    untagged_tokens.append(token)
                elif untagged_tokens:
                    yield self.sub_sequence_for_tokens(untagged_tokens)
                    untagged_tokens = []
            if untagged_tokens:
                yield self.sub_sequence_for_tokens(untagged_tokens)

    def __str__(self):
        return self.tokens_as_str

    def __repr__(self):
        return '{}({})'.format('SequenceWrapper', self.tokens_as_str)


class SequenceWrapperWithPosition(SequenceWrapper):
    def __init__(self, *args, position: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = position

    def sub_sequence_for_tokens(self, tokens: list) -> 'SequenceWrapperWithPosition':
        return SequenceWrapperWithPosition(
            self.structured_document, tokens,
            str_filter_f=self.str_filter_f,
            position=self.position
        )

    def __repr__(self):
        return '{}({}, {})'.format(
            'SequenceWrapperWithPosition', self.tokens_as_str, self.position
        )


class PendingSequences:
    def __init__(self, sequences: Sequence[SequenceWrapper]):
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
        pending_sequences: List[SequenceWrapper] = []
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


class SequencesText:
    def __init__(self, sequences: List[SequenceWrapper], sep: str = '\n', pad: str = ''):
        self._joined_text = JoinedText(sequences, sep=sep, pad=pad)

    @property
    def end_index(self):
        return self._joined_text.end_index

    def iter_sequences_between(self, index_range: Tuple[int, int]) -> Iterable[SequenceWrapper]:
        seq_and_index_range_iterable = self._joined_text.iter_items_and_index_range_between(
            index_range
        )
        for seq, _ in seq_and_index_range_iterable:
            yield seq

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
