import logging
from typing import Set

from sciencebeam_gym.structured_document import AbstractStructuredDocument
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)


LOGGER = logging.getLogger(__name__)


class FrontTagNames:
    TITLE = 'title'


class SegmentationTagNames:
    FRONT = 'front'
    BODY = 'body'


def _get_class_tag_names(c) -> Set[str]:
    return {
        value
        for key, value in c.__dict__.items()
        if key.isupper()
    }


FRONT_TAG_NAMES = _get_class_tag_names(FrontTagNames)


class SegmentationAnnotator(AbstractAnnotator):
    def annotate(self, structured_document: AbstractStructuredDocument):
        for page in structured_document.get_pages():
            for line in structured_document.get_lines_of_page(page):
                for token in structured_document.get_all_tokens_of_line(line):
                    tag_name = structured_document.get_tag(token)
                    LOGGER.debug('token: %s (%s)', token, tag_name)
                    if tag_name in FRONT_TAG_NAMES:
                        LOGGER.debug('front token: %s (%s)', token, tag_name)
                        structured_document.set_tag(token, SegmentationTagNames.FRONT)
                    else:
                        structured_document.set_tag(token, SegmentationTagNames.BODY)
        return structured_document
