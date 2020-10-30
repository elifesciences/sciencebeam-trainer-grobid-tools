import logging
from typing import Callable

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_gym.structured_document import (
    add_tag_prefix,
    I_TAG_PREFIX
)

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.utils import (
    iter_all_tokens_excluding_space
)


LOGGER = logging.getLogger(__name__)


class MergeGroupTagsAnnotatorConfig:
    def __init__(
            self,
            get_group_tag_for_tag_fn: Callable[[str], str],
            tag_level: str = None):
        self.get_group_tag_for_tag_fn = get_group_tag_for_tag_fn
        self.tag_level = tag_level


class MergeGroupTagsPostProcessingAnnotator(AbstractAnnotator):
    def __init__(self, config: MergeGroupTagsAnnotatorConfig):
        self.config = config
        super().__init__()

    def annotate(self, structured_document: GrobidTrainingTeiStructuredDocument):
        # There is currently no support for more than two level tagging,
        # which would allow the a parent level to be represented.
        # As a workaround we are adding the a separate parent tag, to the tokens without tag.
        # That way those tokens will share a common parent element in the output.
        tag_level = self.config.tag_level
        all_tokens_iterable = iter_all_tokens_excluding_space(structured_document)
        unmatched_tags = set()
        current_group_tag = None
        for token in all_tokens_iterable:
            if tag_level:
                # if we looking at sub tags, then only consider tokens with a tag
                if not structured_document.get_tag_or_preserved_tag(token):
                    continue
            tag = structured_document.get_tag_or_preserved_tag_value(token, level=tag_level)
            if tag:
                current_group_tag = self.config.get_group_tag_for_tag_fn(tag)
                if not current_group_tag:
                    unmatched_tags.add(tag)
                continue
            if not current_group_tag:
                continue
            structured_document.set_tag_only(
                token,
                add_tag_prefix(current_group_tag, prefix=I_TAG_PREFIX),
                level=tag_level
            )
            LOGGER.debug('updated group token (%r): %s', current_group_tag, token)
        LOGGER.debug('ignored unmatched tags: %s', unmatched_tags)
        return structured_document
