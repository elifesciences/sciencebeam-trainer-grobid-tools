import logging
from typing import Iterable, Any

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.simple_matching_annotator import (
    SimpleMatchingAnnotator
)


LOGGER = logging.getLogger(__name__)


def _iter_all_tokens(
        structured_document: AbstractStructuredDocument) -> Iterable[Any]:
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )


class SubTagOnlyAnnotator(SimpleMatchingAnnotator):
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
            if not self.config.preserve_sub_annotations:
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
