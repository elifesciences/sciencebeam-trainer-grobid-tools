import logging
import re
from itertools import groupby
from typing import Dict, List, Optional, Set, Iterable, Any, Tuple

from sciencebeam_trainer_grobid_tools.utils.typing import T

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument,
    split_tag_prefix,
    strip_tag_prefix,
    add_tag_prefix,
    B_TAG_PREFIX
)
from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.utils.misc import get_safe

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.annotation.simple_matching_annotator import (
    get_extended_line_token_tags,
    to_inside_tag
)

from sciencebeam_trainer_grobid_tools.annotation.matching_utils import (
    JoinedText
)


LOGGER = logging.getLogger(__name__)


DEFAULT_IDNO_PREFIX_REGEX = r'\b[a-zA-Z]{2,}(\s?:)?$'


class ReferenceAnnotatorConfig:
    def __init__(
            self,
            sub_tag_map: Dict[str, str],
            merge_enabled_sub_tags: Set[str],
            include_prefix_enabled_sub_tags: Set[str],
            include_suffix_enabled_sub_tags: Set[str],
            prefix_regex_by_sub_tag_map: Dict[str, str],
            etal_sub_tag: str,
            etal_merge_enabled_sub_tags: Set[str]):
        self.sub_tag_map = sub_tag_map
        self.merge_enabled_sub_tags = merge_enabled_sub_tags
        self.include_prefix_enabled_sub_tags = include_prefix_enabled_sub_tags
        self.include_suffix_enabled_sub_tags = include_suffix_enabled_sub_tags
        self.prefix_regex_by_sub_tag_map = prefix_regex_by_sub_tag_map
        self.etal_sub_tag = etal_sub_tag
        self.etal_merge_enabled_sub_tags = etal_merge_enabled_sub_tags

    def __repr__(self):
        return '%s(%s)' % (type(self), self.__dict__)


def _iter_all_tokens(
        structured_document: AbstractStructuredDocument) -> Iterable[Any]:
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )


def _iter_group_tokens_by_tag_entity(
        structured_document: AbstractStructuredDocument,
        tokens: Iterable[T]) -> Iterable[Tuple[Optional[str], List[T]]]:
    pending_tag_value = None
    pending_tokens = None
    for token in tokens:
        current_full_tag = structured_document.get_tag(token)
        current_tag_prefix, current_tag_value = split_tag_prefix(current_full_tag)
        if (
            pending_tokens
            and (
                pending_tag_value != current_tag_value
                or current_tag_prefix == B_TAG_PREFIX
            )
        ):
            yield pending_tag_value, pending_tokens
            pending_tokens = None
        if not pending_tokens:
            pending_tag_value = current_tag_value
            pending_tokens = [token]
            continue
        pending_tokens.append(token)
    if pending_tokens:
        yield pending_tag_value, pending_tokens


def _map_tag(tag: str, tag_map: Dict[str, str]) -> str:
    prefix, tag_value = split_tag_prefix(tag)
    return add_tag_prefix(
        tag=tag_map.get(tag_value, tag_value) if tag_value else tag_value,
        prefix=prefix
    )


def _map_tags(tags: List[str], tag_map: Dict[str, str]) -> List[str]:
    return [
        _map_tag(tag, tag_map=tag_map)
        for tag in tags
    ]


def get_prefix_extended_token_tags(
        token_tags: List[str],
        token_texts: List[str],
        prefix_regex_by_tag_map: Dict[str, str],
        token_whitespaces: List[str] = None,
        enabled_tags: Set[str] = None) -> List[Optional[str]]:
    result: List[Optional[str]] = []
    if token_whitespaces is None:
        token_whitespaces = [' '] * len(token_texts)
    _enabled_tags = (
        enabled_tags
        if enabled_tags is not None
        else prefix_regex_by_tag_map.keys()
    )
    grouped_token_tags: List[List[Tuple[Optional[str], str, Optional[str]]]] = [
        list(group)
        for _, group in groupby(
            zip(token_tags, token_texts, token_whitespaces),
            key=lambda pair: strip_tag_prefix(pair[0])
        )
    ]
    LOGGER.debug('grouped_token_tags=%s', grouped_token_tags)
    for index, group in enumerate(grouped_token_tags):
        LOGGER.debug('group: unpacked=%s', group)
        group_tags: List[str]
        group_texts: List[str]
        group_whitespaces: Optional[List[str]]
        group_tags, group_texts, group_whitespaces = zip(*group)  # type: ignore
        LOGGER.debug(
            'group: tags=%s, texts=%s, whitespace=%s',
            group_tags, group_texts, group_whitespaces
        )
        first_group_tag = group_tags[0]
        next_group = grouped_token_tags[index + 1] if index + 1 < len(grouped_token_tags) else None
        first_next_tag = get_safe(get_safe(next_group, 0), 0)
        first_next_prefix, first_next_tag_value = split_tag_prefix(first_next_tag)
        if first_group_tag or first_next_tag_value not in _enabled_tags:
            result.extend(group_tags)
            continue
        assert first_next_tag_value is not None
        joined_text = JoinedText(group_texts, sep=' ', whitespace_list=group_whitespaces)
        prefix_regex = prefix_regex_by_tag_map[first_next_tag_value]
        m = re.search(prefix_regex, str(joined_text))
        LOGGER.debug('m: %s', m)
        if not m:
            result.extend(group_tags)
            continue
        LOGGER.debug('start: %s (%r)', m.start(), str(joined_text)[m.start():])
        matching_tokens = list(joined_text.iter_items_and_index_range_between(
            (m.start(), len(str(joined_text)))
        ))
        LOGGER.debug('matching_tokens: %s', matching_tokens)
        if not matching_tokens:
            result.extend(group_tags)
            continue
        unmatched_token_count = len(group_tags) - len(matching_tokens)
        result.extend([None] * unmatched_token_count)
        result.extend([first_next_tag])
        result.extend([to_inside_tag(first_next_tag)] * (len(matching_tokens) - 1))
        if first_next_prefix == B_TAG_PREFIX:
            assert next_group is not None
            next_group[0] = (
                to_inside_tag(first_next_tag),
                *next_group[0][1:]
            )
    LOGGER.debug('result: %s', result)
    return result


def _add_idno_text_prefix(
        structured_document: GrobidTrainingTeiStructuredDocument,
        tokens: List[Any],
        config: ReferenceAnnotatorConfig):
    sub_tags = [structured_document.get_sub_tag(token) for token in tokens]
    token_texts = [structured_document.get_text(token) for token in tokens]
    token_whitespaces = [structured_document.get_whitespace(token) for token in tokens]
    mapped_sub_tags = _map_tags(sub_tags, config.sub_tag_map)
    transformed_sub_tags = get_prefix_extended_token_tags(
        mapped_sub_tags,
        token_texts,
        prefix_regex_by_tag_map=config.prefix_regex_by_sub_tag_map,
        token_whitespaces=token_whitespaces,
        enabled_tags=config.include_prefix_enabled_sub_tags
    )
    LOGGER.debug(
        'idno prefix sub tokens, transformed: %s -> %s -> %s (tokens: %s)',
        sub_tags, mapped_sub_tags, transformed_sub_tags, tokens
    )
    for token, token_sub_tag in zip(tokens, transformed_sub_tags):
        if not token_sub_tag:
            continue
        structured_document.set_sub_tag(token, token_sub_tag)
    return structured_document


def get_suffix_extended_token_tags(
        token_tags: List[str],
        token_texts: List[str],
        enabled_tags: Set[str],
        token_whitespaces: Optional[List[str]] = None) -> List[Optional[str]]:
    result: List[Optional[str]] = []
    if token_whitespaces is None:
        token_whitespaces = [' '] * len(token_texts)
    grouped_token_tags: List[List[Tuple[str, str, Optional[str]]]] = [
        list(group)
        for _, group in groupby(
            zip(token_tags, token_texts, token_whitespaces),
            key=lambda pair: strip_tag_prefix(pair[0])
        )
    ]
    LOGGER.debug('suffix grouped_token_tags=%s', grouped_token_tags)
    for index, group in enumerate(grouped_token_tags):
        LOGGER.debug('suffix group: unpacked=%s', group)
        group_tags: List[str]
        group_texts: List[str]
        group_whitespaces: Optional[List[str]]
        group_tags, group_texts, group_whitespaces = zip(*group)  # type: ignore
        LOGGER.debug(
            'suffix group: tags=%s, texts=%s, whitespace=%s',
            group_tags, group_texts, group_whitespaces
        )
        first_group_tag = group_tags[0]

        prev_group = grouped_token_tags[index - 1] if index > 0 else None
        first_prev_tag: Optional[str] = get_safe(get_safe(prev_group, 0), 0)
        _, first_prev_tag_value = split_tag_prefix(first_prev_tag)

        if first_group_tag or first_prev_tag_value not in enabled_tags:
            result.extend(group_tags)
            continue
        joined_text = JoinedText(group_texts, sep=' ', whitespace_list=group_whitespaces)
        m = re.search(r'^\.', str(joined_text))
        LOGGER.debug('suffix match: %s', m)
        if not m:
            result.extend(group_tags)
            continue
        LOGGER.debug('suffix match end: %s (%r)', m.end(), str(joined_text)[:m.end()])
        matching_tokens = list(joined_text.iter_items_and_index_range_between(
            (0, m.end())
        ))
        LOGGER.debug('suffix matching_tokens: %s', matching_tokens)
        if not matching_tokens:
            result.extend(group_tags)
            continue
        unmatched_token_count = len(group_tags) - len(matching_tokens)
        result.extend([to_inside_tag(first_prev_tag)] * len(matching_tokens))
        result.extend([None] * unmatched_token_count)
    LOGGER.debug('suffix result: %s', result)
    return result


def _add_name_text_suffix(
        structured_document: GrobidTrainingTeiStructuredDocument,
        tokens: List[Any],
        config: ReferenceAnnotatorConfig):
    sub_tags = [structured_document.get_sub_tag(token) for token in tokens]
    token_texts = [structured_document.get_text(token) for token in tokens]
    token_whitespaces = [structured_document.get_whitespace(token) for token in tokens]
    mapped_sub_tags = _map_tags(sub_tags, config.sub_tag_map)
    transformed_sub_tags = get_suffix_extended_token_tags(
        mapped_sub_tags,
        token_texts,
        token_whitespaces=token_whitespaces,
        enabled_tags=config.include_suffix_enabled_sub_tags
    )
    LOGGER.debug(
        'name suffix sub tokens, transformed: %s -> %s -> %s (tokens: %s)',
        sub_tags, mapped_sub_tags, transformed_sub_tags, tokens
    )
    for token, token_sub_tag in zip(tokens, transformed_sub_tags):
        if not token_sub_tag:
            continue
        structured_document.set_sub_tag(token, token_sub_tag)
    return structured_document


def get_etal_mapped_tags(
        token_tags: List[str],
        etal_sub_tag: str,
        etal_merge_enabled_sub_tags: Set[str]) -> List[str]:
    grouped_token_tags = [
        list(group)
        for _, group in groupby(token_tags, key=strip_tag_prefix)
    ]
    LOGGER.debug('grouped_token_tags: %s', grouped_token_tags)
    result = []
    previous_accepted_group_sub_tag = None
    for group in grouped_token_tags:
        group_tag = group[0]
        group_tag_value = strip_tag_prefix(group_tag)
        if group_tag_value != etal_sub_tag or not previous_accepted_group_sub_tag:
            result.extend(group)
            if group_tag_value in etal_merge_enabled_sub_tags:
                previous_accepted_group_sub_tag = group_tag
            elif group_tag:
                previous_accepted_group_sub_tag = None
            continue
        result.append(previous_accepted_group_sub_tag)
        result.extend(
            [to_inside_tag(previous_accepted_group_sub_tag)]
            * (len(group) - 1)
        )
    return result


def _map_etal_sub_tag(
        structured_document: AbstractStructuredDocument,
        tokens: List[Any],
        config: ReferenceAnnotatorConfig):
    sub_tags = [structured_document.get_sub_tag(token) for token in tokens]
    mapped_sub_tags = _map_tags(sub_tags, config.sub_tag_map)
    transformed_sub_tags = get_etal_mapped_tags(
        mapped_sub_tags,
        etal_sub_tag=config.etal_sub_tag,
        etal_merge_enabled_sub_tags=config.etal_merge_enabled_sub_tags
    )
    LOGGER.debug(
        'etal sub tokens, transformed: %s -> %s -> %s (tokens: %s)',
        sub_tags, mapped_sub_tags, transformed_sub_tags, tokens
    )
    for token, token_sub_tag in zip(tokens, transformed_sub_tags):
        if not token_sub_tag:
            continue
        structured_document.set_sub_tag(token, token_sub_tag)
    return structured_document


def _merge_sub_tags(
        structured_document: AbstractStructuredDocument,
        tokens: List[Any],
        config: ReferenceAnnotatorConfig):
    sub_tags = [structured_document.get_sub_tag(token) for token in tokens]
    mapped_sub_tags = _map_tags(sub_tags, config.sub_tag_map)
    transformed_sub_tags = get_extended_line_token_tags(
        mapped_sub_tags,
        extend_to_line_enabled_map={},
        merge_enabled_map={
            key: True
            for key in config.merge_enabled_sub_tags
        },
        default_merge_enabled=False,
        default_extend_to_line_enabled=False
    )
    LOGGER.debug(
        'sub tokens, transformed: %s -> %s -> %s (tokens: %s)',
        sub_tags, mapped_sub_tags, transformed_sub_tags, tokens
    )
    for token, token_sub_tag in zip(tokens, transformed_sub_tags):
        if not token_sub_tag:
            continue
        structured_document.set_sub_tag(token, token_sub_tag)
    return structured_document


class ReferencePostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ReferenceAnnotatorConfig):
        super().__init__()
        self.config = config

    def annotate(self, structured_document: AbstractStructuredDocument):
        assert isinstance(structured_document, GrobidTrainingTeiStructuredDocument)
        all_tokens_iterable = _iter_all_tokens(structured_document)
        grouped_entity_tokens_iterable = _iter_group_tokens_by_tag_entity(
            structured_document,
            all_tokens_iterable
        )
        for entity_tag_value, entity_tokens in grouped_entity_tokens_iterable:
            LOGGER.debug('entity_tokens (%s): %s', entity_tag_value, entity_tokens)
            _map_etal_sub_tag(
                structured_document,
                entity_tokens,
                config=self.config
            )
            _add_name_text_suffix(
                structured_document,
                entity_tokens,
                config=self.config
            )
            _add_idno_text_prefix(
                structured_document,
                entity_tokens,
                config=self.config
            )
            _merge_sub_tags(
                structured_document,
                entity_tokens,
                config=self.config
            )
        return structured_document
