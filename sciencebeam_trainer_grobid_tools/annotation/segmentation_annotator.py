import logging
from configparser import ConfigParser
from collections import Counter
from typing import Dict, List, Set

from sciencebeam_utils.utils.string import parse_list

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    strip_tag_prefix
)
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)
from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)


LOGGER = logging.getLogger(__name__)


class FrontTagNames:
    TITLE = 'title'
    PAGE = 'page'


class BackTagNames:
    REFERENCE = 'reference'


class SegmentationTagNames:
    FRONT = 'front'
    PAGE = 'page'
    BODY = 'body'
    REFERENCE = 'reference'


def _get_class_tag_names(c) -> Set[str]:
    return {
        value
        for key, value in c.__dict__.items()
        if key.isupper()
    }


FRONT_TAG_NAMES = _get_class_tag_names(FrontTagNames)


DEFAULT_FRONT_MAX_START_LINE = 30


class SegmentationConfig:
    def __init__(
            self,
            segmentation_mapping: Dict[str, Set[str]],
            front_max_start_line_index: int = DEFAULT_FRONT_MAX_START_LINE):
        self.segmentation_mapping = segmentation_mapping
        self.front_max_start_line_index = front_max_start_line_index

    def __repr__(self):
        return '%s(%s)' % (type(self), self.__dict__)


def parse_segmentation_config(filename: str) -> SegmentationConfig:
    with open(filename, 'r') as f:
        config = ConfigParser()
        config.read_file(f)  # pylint: disable=no-member
        front_max_start_line_index = config.getint(
            'config', 'front_max_start_line_index',
            fallback=DEFAULT_FRONT_MAX_START_LINE
        )
        return SegmentationConfig(segmentation_mapping={
            key: set(parse_list(value))
            for key, value in config.items('tags')
        }, front_max_start_line_index=front_max_start_line_index)


def _iter_all_lines(structured_document: AbstractStructuredDocument):
    return (
        line
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
    )


def _set_line_tokens_tag(
        structured_document: AbstractStructuredDocument,
        line,
        tag: str):
    for token in structured_document.get_all_tokens_of_line(line):
        structured_document.set_tag(token, tag)


def _get_line_token_tags(structured_document: AbstractStructuredDocument, line) -> List[str]:
    return [
        structured_document.get_tag(token)
        for token in structured_document.get_tokens_of_line(line)
    ]


def _to_tag_values(tags: List[str]) -> List[str]:
    return list(map(strip_tag_prefix, tags))


def _get_line_token_tag_values(*args, **kwargs) -> List[str]:
    return _to_tag_values(_get_line_token_tags(*args, **kwargs))


def _get_line_token_tags_or_preserved_tags(
        structured_document: GrobidTrainingTeiStructuredDocument, line) -> List[str]:
    return [
        structured_document.get_tag_or_preserved_tag(token)
        for token in structured_document.get_tokens_of_line(line)
    ]


def _clear_line_token_tags(
        structured_document: GrobidTrainingTeiStructuredDocument, line) -> List[str]:
    for token in structured_document.get_all_tokens_of_line(line):
        tag = structured_document.get_tag(token)
        if tag:
            structured_document.set_tag(token, None)


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
        untagged_indexed_lines = []
        min_max_by_tag = {}
        for line_index, line in enumerate(_iter_all_lines(structured_document)):
            full_line_token_tags = _get_line_token_tags(structured_document, line)
            line_token_tags = _to_tag_values(full_line_token_tags)
            line_tag_counts = Counter(line_token_tags)
            if not line_tag_counts:
                continue
            majority_tag_name = line_tag_counts.most_common(1)[0][0]
            segmentation_tag = self.segmentation_tag_name_by_tag_name.get(majority_tag_name)
            LOGGER.debug(
                'line_tag_counts: %s (%s -> %s)',
                line_tag_counts, majority_tag_name, segmentation_tag
            )

            if (
                segmentation_tag == SegmentationTagNames.FRONT
                and segmentation_tag not in min_max_by_tag
                and self.config.front_max_start_line_index
                and line_index > self.config.front_max_start_line_index
            ):
                LOGGER.debug(
                    'ignore front tag beyond line index threshold (%d > %d)',
                    line_index, self.config.front_max_start_line_index
                )
                segmentation_tag = None
                _clear_line_token_tags(structured_document, line)

            if segmentation_tag and segmentation_tag == majority_tag_name:
                LOGGER.debug(
                    'keep line tokens for %s',
                    segmentation_tag
                )
            elif segmentation_tag:
                if segmentation_tag not in min_max_by_tag:
                    min_max_by_tag[segmentation_tag] = [line_index, line_index]
                else:
                    min_max_by_tag[segmentation_tag][1] = line_index
                _set_line_tokens_tag(structured_document, line, segmentation_tag)
            else:
                if majority_tag_name is None:
                    _clear_line_token_tags(structured_document, line)
                    untagged_indexed_lines.append((line_index, line))

        if SegmentationTagNames.FRONT in min_max_by_tag:
            front_min_line_index, front_max_line_index = min_max_by_tag[SegmentationTagNames.FRONT]
            LOGGER.debug(
                'processing untagged lines, front (%d -> %d)',
                front_min_line_index, front_max_line_index
            )
            for line_index, line in list(untagged_indexed_lines):
                tags = _get_line_token_tags_or_preserved_tags(structured_document, line)
                if self.preserve_tags and SegmentationTagNames.PAGE in tags:
                    continue
                if line_index <= front_max_line_index:
                    LOGGER.debug(
                        'tagging as front, within range (%d: %d -> %d)',
                        line_index, front_min_line_index, front_max_line_index
                    )
                    _set_line_tokens_tag(structured_document, line, SegmentationTagNames.FRONT)
                    untagged_indexed_lines.remove((line_index, line))

        if not self.preserve_tags:
            for line_index, line in untagged_indexed_lines:
                _set_line_tokens_tag(structured_document, line, SegmentationTagNames.BODY)
        return structured_document
