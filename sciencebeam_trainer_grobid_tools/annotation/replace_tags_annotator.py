import logging
from typing import Any, Dict, Iterable

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    split_tag_prefix,
    add_tag_prefix
)

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.utils import (
    iter_all_tokens_excluding_space
)


LOGGER = logging.getLogger(__name__)


class ReplaceTagsAnnotatorConfig:
    def __init__(
            self,
            replaced_tag_by_tag: Dict[str, str]):
        self.replaced_tag_by_tag = replaced_tag_by_tag


class ReplaceTagsPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ReplaceTagsAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        ignored_token_tag_values = set()
        for token in all_tokens_iterable:
            tag = structured_document.get_tag_or_preserved_tag(token)
            tag_prefix, tag_value = split_tag_prefix(tag)
            if tag_value not in self.config.replaced_tag_by_tag:
                ignored_token_tag_values.add(tag_value)
                continue
            replaced_tag_value = self.config.replaced_tag_by_tag[tag_value]
            structured_document.set_tag(
                token,
                add_tag_prefix(replaced_tag_value, tag_prefix)
            )
        LOGGER.debug('ignore not enabled tag values: %s', ignored_token_tag_values)
        return structured_document
