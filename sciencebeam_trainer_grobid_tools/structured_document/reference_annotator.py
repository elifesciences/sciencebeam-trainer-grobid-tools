import logging
from typing import Dict, List, Set, Iterable, Any

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    split_tag_prefix,
    add_tag_prefix,
    B_TAG_PREFIX
)
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)
from sciencebeam_trainer_grobid_tools.structured_document.simple_matching_annotator import (
    get_extended_line_token_tags
)


LOGGER = logging.getLogger(__name__)


class ReferenceAnnotatorConfig:
    def __init__(
            self,
            sub_tag_map: Dict[str, str],
            merge_enabled_sub_tags: Set[str]):
        self.sub_tag_map = sub_tag_map
        self.merge_enabled_sub_tags = merge_enabled_sub_tags

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
        }
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
        all_tokens_iterable = _iter_all_tokens(structured_document)
        grouped_entity_tokens_iterable = _iter_group_tokens_by_tag_entity(
            structured_document,
            all_tokens_iterable
        )
        for entity_tag_value, entity_tokens in grouped_entity_tokens_iterable:
            LOGGER.debug('entity_tokens (%s): %s', entity_tag_value, entity_tokens)
            _merge_sub_tags(
                structured_document,
                entity_tokens,
                config=self.config
            )
        return structured_document
