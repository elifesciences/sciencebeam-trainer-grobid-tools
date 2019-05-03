import logging
from mock import MagicMock

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)

from sciencebeam_trainer_grobid_tools.structured_document.annotator import (
    annotate_structured_document_inplace
)

from .grobid_training_tei_test import _tei


LOGGER = logging.getLogger(__name__)


TITLE_1 = 'Title 1'


@pytest.fixture(name='annotator')
def _annotator():
    return MagicMock(name='annotator')


def _tei_with_title(title=TITLE_1):
    return _tei(front_items=[
        E.docTitle(E.titlePart(title))
    ])


def _structured_document_with_title(title=TITLE_1):
    return GrobidTrainingTeiStructuredDocument(_tei_with_title(title))


def _get_root(structured_document):
    root = structured_document.root
    LOGGER.debug('root: %s', etree.tostring(root))
    return root


class TestAnnotateStructuredDocumentInplace(object):
    def test_should_not_preserve_tags(self, annotator):
        structured_document = _structured_document_with_title()
        assert len(_get_root(structured_document).xpath('//docTitle')) == 1

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=False,
            fields=['other']
        )
        assert len(_get_root(structured_document).xpath('//docTitle')) == 0

    def test_should_preserve_tags(self, annotator):
        structured_document = _structured_document_with_title()
        assert len(_get_root(structured_document).xpath('//docTitle')) == 1

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=True,
            fields=['other']
        )
        assert len(_get_root(structured_document).xpath('//docTitle')) == 1

    def test_should_not_preserve_tags_of_fields(self, annotator):
        structured_document = _structured_document_with_title()
        assert len(_get_root(structured_document).xpath('//docTitle')) == 1

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=True,
            fields=['title']
        )
        assert len(_get_root(structured_document).xpath('//docTitle')) == 0
