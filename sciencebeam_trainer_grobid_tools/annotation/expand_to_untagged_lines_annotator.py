import logging
from typing import Any, List, Set

from sciencebeam_gym.structured_document import (
    I_TAG_PREFIX
)

from sciencebeam_gym.preprocess.annotation.annotator import (
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


def _log_previous_included_tokens(
        previous_enabled_tag_value: str,
        previous_included_tokens: List[Any]):
    if previous_included_tokens:
        LOGGER.debug(
            'included untagged tokens in previous tag %r: %s',
            previous_enabled_tag_value, previous_included_tokens
        )


class ExpandToUntaggedLinesPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ExpandToUntaggedLinesAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        previous_enabled_tag_value = None
        previous_included_tokens = []
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
            structured_document.set_tag(
                token,
                add_tag_prefix(previous_enabled_tag_value, I_TAG_PREFIX)
            )
            previous_included_tokens.append(token)
        _log_previous_included_tokens(
            previous_enabled_tag_value, previous_included_tokens
        )
        LOGGER.debug('ignore not enabled tag values: %s', ignored_token_tag_values)
        return structured_document
