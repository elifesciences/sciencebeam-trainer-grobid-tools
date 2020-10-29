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


LOGGER = logging.getLogger(__name__)


class ReplaceTagsAnnotatorConfig:
    def __init__(
            self,
            replaced_tag_by_tag: Dict[str, str]):
        self.replaced_tag_by_tag = replaced_tag_by_tag


def _iter_all_tokens(
        structured_document: AbstractStructuredDocument) -> Iterable[Any]:
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )


class ReplaceTagsPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: ReplaceTagsAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        all_tokens_iterable = _iter_all_tokens(structured_document)
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
