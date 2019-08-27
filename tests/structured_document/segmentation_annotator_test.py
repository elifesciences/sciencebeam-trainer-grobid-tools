import logging
from pathlib import Path
from typing import List, Tuple

from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    ContainerNodePaths
)

from sciencebeam_trainer_grobid_tools.structured_document.segmentation_annotator import (
    parse_segmentation_config,
    SegmentationConfig,
    SegmentationAnnotator,
    FrontTagNames,
    SegmentationTagNames
)


LOGGER = logging.getLogger(__name__)


SEGMENTATION_CONTAINER_NODE_PATH = ContainerNodePaths.SEGMENTATION_CONTAINER_NODE_PATH


DEFAULT_CONFIG = SegmentationConfig({
    SegmentationTagNames.FRONT: {FrontTagNames.TITLE}
})


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'


OTHER_TAG = 'other'


def _tei(items: list = None):
    return E.tei(E.text(
        *(items or [])
    ))


def _simple_document_with_tagged_token_lines(
        lines: List[List[Tuple[str, str]]]) -> GrobidTrainingTeiStructuredDocument:
    tei_items = []
    for line in lines:
        tei_items.append(' '.join(token for _, token in line))
        tei_items.append(E.lb())
    doc = GrobidTrainingTeiStructuredDocument(
        _tei(tei_items),
        container_node_path=SEGMENTATION_CONTAINER_NODE_PATH
    )
    doc_lines = [line for page in doc.get_pages() for line in doc.get_lines_of_page(page)]
    for line, doc_line in zip(lines, doc_lines):
        for (tag, token), doc_token in zip(line, doc.get_tokens_of_line(doc_line)):
            assert token == doc.get_text(doc_token)
            if tag:
                doc.set_tag(doc_token, tag)
    return doc


def _get_document_tagged_token_lines(
        doc: GrobidTrainingTeiStructuredDocument) -> List[List[Tuple[str, str]]]:
    document_tagged_token_lines = [
        [
            (doc.get_tag(token), doc.get_text(token))
            for token in doc.get_tokens_of_line(line)
        ]
        for page in doc.get_pages()
        for line in doc.get_lines_of_page(page)
    ]
    LOGGER.debug('document_tagged_token_lines: %s', document_tagged_token_lines)
    return document_tagged_token_lines


class TestParseSegmentationConfig:
    def test_should_parse_config(self, temp_dir: Path):
        config_path = temp_dir.joinpath('segmentation.conf')
        config_path.write_text('\n'.join([
            '[tags]',
            'front = title, abstract '
        ]))
        config = parse_segmentation_config(config_path)
        LOGGER.debug('config: %s', config)
        assert config.segmentation_mapping['front'] == {'title', 'abstract'}


class TestSegmentationAnnotator:
    def test_should_not_fail_on_empty_document(self):
        structured_document = GrobidTrainingTeiStructuredDocument(
            _tei()
        )
        SegmentationAnnotator(DEFAULT_CONFIG).annotate(structured_document)

    def test_should_annotate_title_as_front(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1)
            ]
        ]

    def test_should_annotate_other_tags_as_body(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(OTHER_TAG, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.BODY, TOKEN_1)
            ]
        ]

    def test_should_annotate_no_tag_as_body_if_preserve_is_disabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=False).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.BODY, TOKEN_1)
            ]
        ]

    def test_should_annotate_not_no_tag_as_body_if_preserve_is_enabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (None, TOKEN_1)
            ]
        ]

    def test_should_annotate_not_fail_on_empty_line(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [],
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [],
            [(None, TOKEN_1)]
        ]

    def test_should_annotate_title_line_as_front(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (FrontTagNames.TITLE, TOKEN_1),
                (FrontTagNames.TITLE, TOKEN_2),
                (FrontTagNames.TITLE, TOKEN_3)
            ]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1),
                (SegmentationTagNames.FRONT, TOKEN_2),
                (SegmentationTagNames.FRONT, TOKEN_3)
            ]
        ]

    def test_should_annotate_line_with_using_common_tag(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (FrontTagNames.TITLE, TOKEN_1),
                (FrontTagNames.TITLE, TOKEN_2),
                (OTHER_TAG, TOKEN_3)
            ]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1),
                (SegmentationTagNames.FRONT, TOKEN_2),
                (SegmentationTagNames.FRONT, TOKEN_3)
            ]
        ]

    def test_should_annotate_untagged_lines_between_first_and_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(OTHER_TAG, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_annotate_untagged_lines_before_first_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(OTHER_TAG, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_not_annotate_untagged_lines_after_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(OTHER_TAG, TOKEN_3)],
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(OTHER_TAG, TOKEN_3)]
        ]

    def test_should_not_annotate_untagged_page_no_lines_between_first_and_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(FrontTagNames.PAGE, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.PAGE, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]
