import logging
from configparser import ConfigParser
from collections import Counter
from typing import Dict, Set

from sciencebeam_utils.utils.string import parse_list
from sciencebeam_gym.structured_document import AbstractStructuredDocument
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)


LOGGER = logging.getLogger(__name__)


class FrontTagNames:
    TITLE = 'title'


class SegmentationTagNames:
    FRONT = 'front'
    PAGE = 'page'
    BODY = 'body'


def _get_class_tag_names(c) -> Set[str]:
    return {
        value
        for key, value in c.__dict__.items()
        if key.isupper()
    }


FRONT_TAG_NAMES = _get_class_tag_names(FrontTagNames)


class SegmentationConfig:
    def __init__(self, segmentation_mapping: Dict[str, Set[str]]):
        self.segmentation_mapping = segmentation_mapping

    def __repr__(self):
        return '%s(%s)' % (type(self), self.__dict__)


def parse_segmentation_config(filename: str) -> SegmentationConfig:
    with open(filename, 'r') as f:
        config = ConfigParser()
        config.read_file(f)  # pylint: disable=no-member
        return SegmentationConfig(segmentation_mapping={
            key: set(parse_list(value))
            for key, value in config.items('tags')
        })


class SegmentationAnnotator(AbstractAnnotator):
    def __init__(self, config: SegmentationConfig, preserve_tags: bool = False):
        super().__init__()
        self.config = config
        self.preserve_tags = preserve_tags
        self.segmentation_tag_name_by_tag_name = {
            tag_name: segmentation_tag
            for segmentation_tag, tag_names in config.segmentation_mapping.items()
            for tag_name in tag_names
        }

    def annotate(self, structured_document: AbstractStructuredDocument):
        for page in structured_document.get_pages():
            for line in structured_document.get_lines_of_page(page):
                line_tag_counts = Counter(
                    structured_document.get_tag(token)
                    for token in structured_document.get_tokens_of_line(line)
                )
                majority_tag_name = line_tag_counts.most_common(1)[0][0]
                segmentation_tag = self.segmentation_tag_name_by_tag_name.get(majority_tag_name)
                LOGGER.debug(
                    'line_tag_counts: %s (%s -> %s)',
                    line_tag_counts, majority_tag_name, segmentation_tag
                )
                if not segmentation_tag and not self.preserve_tags:
                    segmentation_tag = SegmentationTagNames.BODY
                if segmentation_tag:
                    for token in structured_document.get_all_tokens_of_line(line):
                        structured_document.set_tag(token, segmentation_tag)
        return structured_document
