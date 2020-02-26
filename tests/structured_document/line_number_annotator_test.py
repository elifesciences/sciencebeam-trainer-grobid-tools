import logging
from typing import List, Tuple

from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    ContainerNodePaths
)

from sciencebeam_trainer_grobid_tools.structured_document.line_number_annotator import (
    TextLineNumberAnnotatorConfig,
    TextLineNumberAnnotator,
    DEFAULT_LINE_NO_TAG as LINE_NO_TAG
)


LOGGER = logging.getLogger(__name__)


DEFAULT_CONTAINER_NODE_PATH = ContainerNodePaths.SEGMENTATION_CONTAINER_NODE_PATH

TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'

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
        container_node_path=DEFAULT_CONTAINER_NODE_PATH
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


class TestTextLineNumberAnnotator:
    def test_should_not_fail_on_empty_document(self):
        structured_document = GrobidTrainingTeiStructuredDocument(
            _tei()
        )
        TextLineNumberAnnotator().annotate(structured_document)

    def test_should_not_annotate_general_token_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, TOKEN_1)]
        ])

        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            line_number_ratio_threshold=0.7
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(TAG_1, TOKEN_1)]
        ]

    def test_should_annotate_sequential_numbers_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, '2'), (TAG_1, TOKEN_2)],
            [(TAG_1, '3'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            line_number_ratio_threshold=0.7
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(LINE_NO_TAG, '1'), (TAG_1, TOKEN_1)],
            [(LINE_NO_TAG, '2'), (TAG_1, TOKEN_2)],
            [(LINE_NO_TAG, '3'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_fail_on_unicode_digit(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, '2'), (TAG_1, TOKEN_2)],
            [(TAG_1, '\u2083'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            line_number_ratio_threshold=0.3
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(LINE_NO_TAG, '1'), (TAG_1, TOKEN_1)],
            [(LINE_NO_TAG, '2'), (TAG_1, TOKEN_2)],
            [(TAG_1, '\u2083'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_annotate_sequential_numbers_with_suffix_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1a'), (TAG_1, TOKEN_1)],
            [(TAG_1, '2a'), (TAG_1, TOKEN_2)],
            [(TAG_1, '3a'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            line_number_ratio_threshold=0.7
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(TAG_1, '1a'), (TAG_1, TOKEN_1)],
            [(TAG_1, '2a'), (TAG_1, TOKEN_2)],
            [(TAG_1, '3a'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_annotate_individual_numbers_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, 'A'), (TAG_1, TOKEN_2)],
            [(TAG_1, 'B'), (TAG_1, TOKEN_3)]
        ])
        TextLineNumberAnnotator().annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, 'A'), (TAG_1, TOKEN_2)],
            [(TAG_1, 'B'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_annotate_sparse_numbers_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, 'A'), (TAG_1, TOKEN_2)],
            [(TAG_1, 'B'), (TAG_1, TOKEN_3)],
            [(TAG_1, '2'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            line_number_ratio_threshold=0.7
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, 'A'), (TAG_1, TOKEN_2)],
            [(TAG_1, 'B'), (TAG_1, TOKEN_3)],
            [(TAG_1, '2'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_annotate_out_of_sequence_numbers_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, '9'), (TAG_1, TOKEN_2)],
            [(TAG_1, '2'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            max_line_number_gap=0,
            line_number_ratio_threshold=0.5
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(LINE_NO_TAG, '1'), (TAG_1, TOKEN_1)],
            [(TAG_1, '9'), (TAG_1, TOKEN_2)],
            [(LINE_NO_TAG, '2'), (TAG_1, TOKEN_3)]
        ]

    def test_should_not_annotate_same_out_of_sequence_numbers_at_line_start_as_line_no(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '1'), (TAG_1, 'token1')],
            [(TAG_1, '2'), (TAG_1, 'token2')],
            [(TAG_1, '3'), (TAG_1, 'token3')],
            [(TAG_1, '4'), (TAG_1, 'token4')],
            [(TAG_1, '1'), (TAG_1, 'out_of_sequence_1')],
            [(TAG_1, '5'), (TAG_1, 'token5')],
            [(TAG_1, '6'), (TAG_1, 'token6')],
            [(TAG_1, '7'), (TAG_1, 'token7')]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            max_line_number_gap=0,
            line_number_ratio_threshold=0.5
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(LINE_NO_TAG, '1'), (TAG_1, 'token1')],
            [(LINE_NO_TAG, '2'), (TAG_1, 'token2')],
            [(LINE_NO_TAG, '3'), (TAG_1, 'token3')],
            [(LINE_NO_TAG, '4'), (TAG_1, 'token4')],
            [(TAG_1, '1'), (TAG_1, 'out_of_sequence_1')],
            [(LINE_NO_TAG, '5'), (TAG_1, 'token5')],
            [(LINE_NO_TAG, '6'), (TAG_1, 'token6')],
            [(LINE_NO_TAG, '7'), (TAG_1, 'token7')]
        ]

    def test_should_annotate_longest_sequence_of_sequential_numbers(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(TAG_1, '10'), (TAG_1, TOKEN_1)],
            [(TAG_1, '11'), (TAG_1, TOKEN_2)],
            [(TAG_1, '1'), (TAG_1, TOKEN_3)],
            [(TAG_1, '2'), (TAG_1, TOKEN_3)],
            [(TAG_1, '3'), (TAG_1, TOKEN_3)]
        ])
        config = TextLineNumberAnnotatorConfig(
            min_line_number=1,
            max_line_number_gap=0,
            line_number_ratio_threshold=0.5
        )
        TextLineNumberAnnotator(config=config).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(TAG_1, '10'), (TAG_1, TOKEN_1)],
            [(TAG_1, '11'), (TAG_1, TOKEN_2)],
            [(LINE_NO_TAG, '1'), (TAG_1, TOKEN_3)],
            [(LINE_NO_TAG, '2'), (TAG_1, TOKEN_3)],
            [(LINE_NO_TAG, '3'), (TAG_1, TOKEN_3)]
        ]
