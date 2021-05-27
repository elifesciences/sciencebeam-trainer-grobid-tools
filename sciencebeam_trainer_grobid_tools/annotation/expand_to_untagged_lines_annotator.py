import logging
from typing import Any, List, Optional, Set

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    B_TAG_PREFIX,
    I_TAG_PREFIX
)

from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    add_tag_prefix,
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.utils import (
    iter_all_tokens_excluding_space
)


LOGGER = logging.getLogger(__name__)


class ExpandToUntaggedLinesAnnotatorConfig:
    def __init__(
            self,
            enabled_tags: Set[str]):
        self.enabled_tags = enabled_tags


class ExpandToPreviousUntaggedLinesPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ExpandToUntaggedLinesAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        previous_untagged_tokens = []
        ignored_token_tag_values = set()
        for token in all_tokens_iterable:
            tag_value = structured_document.get_tag_or_preserved_tag_value(token)
            if not tag_value:
                previous_untagged_tokens.append(token)
                continue
            if not previous_untagged_tokens:
                continue
            if tag_value not in self.config.enabled_tags:
                LOGGER.debug(
                    'ignoring untagged tokens (before %r): %s',
                    tag_value, previous_untagged_tokens
                )
                ignored_token_tag_values.add(tag_value)
                previous_untagged_tokens.clear()
                continue
            LOGGER.debug(
                'updated untagged tokens (before %r): %s',
                tag_value, previous_untagged_tokens
            )
            for index, previous_untagged_token in enumerate(previous_untagged_tokens):
                structured_document.set_tag_only(
                    previous_untagged_token,
                    add_tag_prefix(tag_value, B_TAG_PREFIX if index == 0 else I_TAG_PREFIX)
                )
            structured_document.set_tag_only(
                token,
                add_tag_prefix(tag_value, I_TAG_PREFIX)
            )
            previous_untagged_tokens.clear()
        LOGGER.debug('ignore not enabled tag values: %s', ignored_token_tag_values)
        return structured_document


def _log_previous_included_tokens(
        previous_enabled_tag_value: Optional[str],
        previous_included_tokens: List[Any]):
    if previous_included_tokens:
        LOGGER.debug(
            'included untagged tokens in previous tag %r: %s',
            previous_enabled_tag_value, previous_included_tokens
        )


class ExpandToFollowingUntaggedLinesPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ExpandToUntaggedLinesAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        previous_enabled_tag_value: Optional[str] = None
        previous_included_tokens: List[Any] = []
        ignored_token_tag_values = set()
        for token in all_tokens_iterable:
            tag_value = structured_document.get_tag_or_preserved_tag_value(token)
            if tag_value:
                _log_previous_included_tokens(
                    previous_enabled_tag_value, previous_included_tokens
                )
                previous_included_tokens.clear()
                previous_enabled_tag_value = (
                    tag_value
                    if tag_value in self.config.enabled_tags
                    else None
                )
                if not previous_enabled_tag_value:
                    ignored_token_tag_values.add(tag_value)
                continue
            if not previous_enabled_tag_value:
                continue
            structured_document.set_tag_only(
                token,
                add_tag_prefix(previous_enabled_tag_value, I_TAG_PREFIX)
            )
            previous_included_tokens.append(token)
        _log_previous_included_tokens(
            previous_enabled_tag_value, previous_included_tokens
        )
        LOGGER.debug('ignore not enabled tag values: %s', ignored_token_tag_values)
        return structured_document
