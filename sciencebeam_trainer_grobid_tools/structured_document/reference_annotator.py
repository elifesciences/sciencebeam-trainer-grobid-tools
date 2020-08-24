import logging
import re
from itertools import groupby
from typing import Dict, List, Set, Iterable, Any

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    split_tag_prefix,
    strip_tag_prefix,
    add_tag_prefix,
    B_TAG_PREFIX
)
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.utils.misc import get_safe

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.simple_matching_annotator import (
    get_extended_line_token_tags,
    to_inside_tag,
    SimpleMatchingAnnotator
)

from sciencebeam_trainer_grobid_tools.structured_document.matching_utils import (
    JoinedText
)


LOGGER = logging.getLogger(__name__)


class ReferenceAnnotatorConfig:
    def __init__(
            self,
            sub_tag_map: Dict[str, str],
            merge_enabled_sub_tags: Set[str],
            include_prefix_enabled_sub_tags: Set[str],
            etal_sub_tag: str,
            etal_merge_enabled_sub_tags: Set[str]):
        self.sub_tag_map = sub_tag_map
        self.merge_enabled_sub_tags = merge_enabled_sub_tags
        self.include_prefix_enabled_sub_tags = include_prefix_enabled_sub_tags
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
        tokens: Iterable[Any]) -> Iterable[List[Any]]:
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
        tag=tag_map.get(tag_value, tag_value),
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
        token_whitespaces: List[str] = None,
        enabled_tags: Set[str] = None) -> List[str]:
    result = []
    if token_whitespaces is None:
        token_whitespaces = [' '] * len(token_texts)
    grouped_token_tags = [
        list(group)
        for _, group in groupby(
            zip(token_tags, token_texts, token_whitespaces),
            key=lambda pair: strip_tag_prefix(pair[0])
        )
    ]
    LOGGER.debug('grouped_token_tags=%s', grouped_token_tags)
    for index, group in enumerate(grouped_token_tags):
        LOGGER.debug('group: unpacked=%s', group)
        group_tags, group_texts, group_whitespaces = zip(*group)
        LOGGER.debug(
            'group: tags=%s, texts=%s, whitespace=%s',
            group_tags, group_texts, group_whitespaces
        )
        first_group_tag = group_tags[0]
        next_group = grouped_token_tags[index + 1] if index + 1 < len(grouped_token_tags) else None
        first_next_tag = get_safe(get_safe(next_group, 0), 0)
        first_next_prefix, first_next_tag_value = split_tag_prefix(first_next_tag)
        if first_group_tag or first_next_tag_value not in enabled_tags:
            result.extend(group_tags)
            continue
        joined_text = JoinedText(group_texts, sep=' ', whitespace_list=group_whitespaces)
        m = re.search(r'\b[a-zA-Z]{2,}(\s?:)?$', str(joined_text))
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
        token_whitespaces,
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
        default_extend_to_line_enabled=True
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


class ReferenceSubTagOnlyAnnotator(SimpleMatchingAnnotator):
    def update_annotation_for_index_range(self, *_, **__):  # pylint: disable=arguments-differ
        pass

    def extend_annotations_to_whole_line(self, *_, **__):  # pylint: disable=arguments-differ
        pass

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        LOGGER.debug('preserving tags')
        token_preserved_tags = [
            (token, structured_document.get_tag_or_preserved_tag(token))
            for token in _iter_all_tokens(structured_document)
        ]
        # we need to clear the tag for now, otherwise they will be ignored for annotation
        for token, _ in token_preserved_tags:
            structured_document.set_tag_only(
                token,
                None
            )
            structured_document.clear_preserved_sub_tag(token)
        # process auto-annotations
        super().annotate(structured_document)
        # restore original tags (but now with auto-annotated sub-tags)
        for token, preserved_tag in token_preserved_tags:
            preserved_tag = structured_document.get_tag_or_preserved_tag(token)
            LOGGER.debug('restoring preserved tag: %r -> %r', token, preserved_tag)
            structured_document.set_tag_only(
                token,
                structured_document.get_tag_or_preserved_tag(token)
            )
        return structured_document


class ReferencePostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ReferenceAnnotatorConfig):
        super().__init__()
        self.config = config

    def annotate(self, structured_document: AbstractStructuredDocument):
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
