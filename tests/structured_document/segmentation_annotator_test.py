import logging

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.segmentation_annotator import (
    SegmentationAnnotator,
    FrontTagNames,
    SEGMENTATION_CONTAINER_NODE_PATH
)

from .grobid_training_tei_test import (
    _get_all_lines
)


LOGGER = logging.getLogger(__name__)


TOKEN_1 = 'Token1'


def _tei(items: list = None):
    return E.tei(E.text(
        *(items or [])
    ))


class TestSegmentationAnnotator:
    def test_should_not_fail_on_empty_document(self):
        structured_document = GrobidTrainingTeiStructuredDocument(
            _tei()
        )
        SegmentationAnnotator().annotate(structured_document)

    def test_should_annotate_title_as_front(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(TOKEN_1),
            container_node_path=SEGMENTATION_CONTAINER_NODE_PATH
        )
        lines = _get_all_lines(doc)
        token1 = list(doc.get_tokens_of_line(lines[0]))[0]
        doc.set_tag(token1, FrontTagNames.TITLE)

        SegmentationAnnotator().annotate(doc)
        tei_auto_root = doc.root
        LOGGER.info('tei_auto_root: %s', etree.tostring(tei_auto_root))
        front_nodes = tei_auto_root.xpath('//text/front')
        assert front_nodes
        assert front_nodes[0].text == TOKEN_1

    def test_should_annotate_other_tags_as_body(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(TOKEN_1),
            container_node_path=SEGMENTATION_CONTAINER_NODE_PATH
        )
        lines = _get_all_lines(doc)
        token1 = list(doc.get_tokens_of_line(lines[0]))[0]
        doc.set_tag(token1, 'other')

        SegmentationAnnotator().annotate(doc)
        tei_auto_root = doc.root
        LOGGER.info('tei_auto_root: %s', etree.tostring(tei_auto_root))
        front_nodes = tei_auto_root.xpath('//text/body')
        assert front_nodes
        assert front_nodes[0].text == TOKEN_1
