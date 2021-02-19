import logging
from configparser import ConfigParser
from collections import Counter
from typing import Dict, List, Optional, Set

from sciencebeam_utils.utils.string import parse_list

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument,
    strip_tag_prefix
)
from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)
from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.annotation.matching_utils import join_tokens_text


LOGGER = logging.getLogger(__name__)


class PageTagNames:
    PAGE = 'page'


class FrontTagNames:
    TITLE = 'title'
    ABSTRACT = 'abstract'


class BodyTagNames:
    SECTION_TITLE = 'section_title'


class BackTagNames:
    REFERENCE = 'reference'


class SegmentationTagNames:
    FRONT = 'front'
    PAGE = 'page'
    HEADNOTE = 'headnote'
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


def get_segmentation_tag_name_by_tag_name(
        segmentation_config: SegmentationConfig) -> Dict[str, str]:
    return {
        tag_name: segmentation_tag
        for segmentation_tag, tag_names in segmentation_config.segmentation_mapping.items()
        for tag_name in tag_names
    }


class SegmentationLine:
    def __init__(
        self,
        structured_document: AbstractStructuredDocument,
        line_index: int,
        line,
        segmentation_tag: Optional[str]
    ):
        self.structured_document = structured_document
        self.line_index = line_index
        self.line = line
        self.segmentation_tag = segmentation_tag
        self.text = join_tokens_text(self.structured_document.get_tokens_of_line(line))

    def __repr__(self):
        return '%s(line_index=%d, segmentation_tag=%s, line=%s)' % (
            type(self).__name__, self.line_index, self.segmentation_tag, self.line
        )

    def set_segmentation_tag(self, segmentation_tag: Optional[str]):
        self.segmentation_tag = segmentation_tag
        _set_line_tokens_tag(self.structured_document, self.line, segmentation_tag)

    def clear_line_token_tags(self):
        self.segmentation_tag = None
        _clear_line_token_tags(self.structured_document, self.line)

    def get_line_token_tags(self) -> List[Optional[str]]:
        return _get_line_token_tags(self.structured_document, self.line)


class SegmentationLineList:
    def __init__(self, lines: List[SegmentationLine]):
        self.lines = lines
        self.line_by_line_index_map = {
            line.line_index: line
            for line in lines
        }

    def __iter__(self):
        return iter(self.lines)

    def iter_untagged(self):
        return (line for line in self.lines if not line.segmentation_tag)

    def get_segmentation_tag_by_line_index(self, line_index: int) -> Optional[str]:
        line = self.line_by_line_index_map.get(line_index)
        return line.segmentation_tag if line else None


def find_and_tag_page_headers(
    segmentation_lines: SegmentationLineList
):
    untagged_line_counts = Counter((
        line.text for line in segmentation_lines.iter_untagged()
    ))
    LOGGER.debug('untagged_line_counts: %s', untagged_line_counts)
    if not untagged_line_counts:
        return
    for text, count in untagged_line_counts.most_common():
        if count < 2:
            break
        LOGGER.debug('setting page header (headnote) for line(s): %r (%d)', text, count)
        for line in segmentation_lines:
            if line.text == text:
                line.set_segmentation_tag(SegmentationTagNames.HEADNOTE)


def merge_front_lines(
    segmentation_lines: SegmentationLineList,
    preserve_tags: bool
):
    condidate_lines = []
    previous_segmentation_tag: Optional[str] = SegmentationTagNames.FRONT
    total_merged_lines = 0
    ignored_segmentation_tags = {SegmentationTagNames.HEADNOTE, SegmentationTagNames.PAGE}
    for line in segmentation_lines:
        if line.segmentation_tag in ignored_segmentation_tags:
            continue
        if line.segmentation_tag:
            previous_segmentation_tag = line.segmentation_tag
        if line.segmentation_tag == SegmentationTagNames.FRONT:
            if condidate_lines:
                LOGGER.debug(
                    'tagging as front, merging with previous front line: %s',
                    condidate_lines
                )
                total_merged_lines += len(condidate_lines)
                for condidate_line in condidate_lines:
                    condidate_line.set_segmentation_tag(SegmentationTagNames.FRONT)
            condidate_lines = []
            continue
        if line.segmentation_tag:
            condidate_lines = []
            continue
        tags = _get_line_token_tags_or_preserved_tags(
            line.structured_document, line.line
        )
        if preserve_tags and SegmentationTagNames.PAGE in tags:
            continue
        if previous_segmentation_tag == SegmentationTagNames.FRONT:
            condidate_lines.append(line)
    LOGGER.debug('merged front lines, %d lines', total_merged_lines)


class SegmentationAnnotator(AbstractAnnotator):
    def __init__(self, config: SegmentationConfig, preserve_tags: bool = False):
        super().__init__()
        self.config = config
        self.preserve_tags = preserve_tags
        self.segmentation_tag_name_by_tag_name = get_segmentation_tag_name_by_tag_name(config)

    def annotate(self, structured_document: AbstractStructuredDocument):
        segmentation_lines = SegmentationLineList([
            SegmentationLine(
                structured_document,
                line_index=line_index,
                line=line,
                segmentation_tag=None
            )
            for line_index, line in enumerate(_iter_all_lines(structured_document))
        ])
        for line in segmentation_lines.lines:
            line_index = line.line_index
            full_line_token_tags = line.get_line_token_tags()
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
                and self.config.front_max_start_line_index
                and line_index > self.config.front_max_start_line_index
            ):
                LOGGER.debug(
                    'ignore front tag beyond line index threshold (%d > %d)',
                    line_index, self.config.front_max_start_line_index
                )
                segmentation_tag = None
                line.clear_line_token_tags()

            if segmentation_tag and segmentation_tag == majority_tag_name:
                LOGGER.debug(
                    'keep line tokens for %s',
                    segmentation_tag
                )
            elif segmentation_tag:
                line.set_segmentation_tag(segmentation_tag)
            elif majority_tag_name is None:
                line.clear_line_token_tags()

            line.segmentation_tag = segmentation_tag or majority_tag_name

        find_and_tag_page_headers(segmentation_lines)
        merge_front_lines(
            segmentation_lines=segmentation_lines,
            preserve_tags=self.preserve_tags
        )

        if not self.preserve_tags:
            for line in segmentation_lines.iter_untagged():
                line.set_segmentation_tag(SegmentationTagNames.BODY)
        return structured_document
