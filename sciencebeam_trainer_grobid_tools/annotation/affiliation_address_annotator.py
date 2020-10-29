import logging
from typing import Callable

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    SUB_LEVEL,
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.utils import (
    iter_all_tokens_excluding_space
)


LOGGER = logging.getLogger(__name__)


class AffiliationAddressAnnotatorConfig:
    def __init__(
            self,
            address_sub_tag: str,
            is_address_sub_tag_fn: Callable[[str], bool]):
        self.address_sub_tag = address_sub_tag
        self.is_address_sub_tag_fn = is_address_sub_tag_fn


class AffiliationAddressPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: AffiliationAddressAnnotatorConfig):
        self.config = config
        super().__init__()

    def join_address_fields_by_adding_address_sub_tag(
            self, structured_document: GrobidTrainingTeiStructuredDocument):
        # There is currently no support for three level tagging,
        # which would allow the "address" level to be represented.
        # As a workaround we are adding the "address" sub tag, to the tokens without sub tag.
        # That way those tokens will share a common "address" element in the output.
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        is_in_address = False
        for token in all_tokens_iterable:
            tag = structured_document.get_tag_or_preserved_tag(token)
            if not tag:
                continue
            sub_tag = structured_document.get_tag_or_preserved_tag(token, level=SUB_LEVEL)
            if sub_tag:
                is_in_address = self.config.is_address_sub_tag_fn(sub_tag)
                continue
            if not is_in_address:
                continue
            structured_document.set_tag(token, self.config.address_sub_tag, level=SUB_LEVEL)
            LOGGER.debug('updated address token: %s', token)

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        self.join_address_fields_by_adding_address_sub_tag(structured_document)
        return structured_document
