import logging
from unittest.mock import MagicMock
from typing import List

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.utils.xml import get_xpath_text

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    DEFAULT_TAG_TO_TEI_PATH_MAPPING as _DEFAULT_TAG_TO_TEI_PATH_MAPPING
)

from sciencebeam_trainer_grobid_tools.structured_document.annotator import (
    annotate_structured_document_inplace
)

from .grobid_training_tei_test import _tei


LOGGER = logging.getLogger(__name__)


TITLE_1 = 'Title 1'


TAG_TO_TEI_PATH_MAPPING = {
    **_DEFAULT_TAG_TO_TEI_PATH_MAPPING,
    'parent': 'parent',
    'sub1': 'parent/sub1',
    'sub2': 'parent/sub2'
}


@pytest.fixture(name='annotator')
def _annotator():
    return MagicMock(name='annotator')


def _tei_with_title(title=TITLE_1):
    return _tei(front_items=[
        E.docTitle(E.titlePart(title))
    ])


def _structured_document_with_title(title=TITLE_1):
    return GrobidTrainingTeiStructuredDocument(_tei_with_title(title))


def _tei_with_sub_elements(*sub_elements: List[etree.Element]):
    return _tei(front_items=[
        E.parent(
            'parent-text-to-force-sub-tags ',
            *sub_elements
        )
    ])


def _structured_document_with_sub_elements(*args, **kwargs):
    return GrobidTrainingTeiStructuredDocument(
        _tei_with_sub_elements(*args, **kwargs),
        tag_to_tei_path_mapping=TAG_TO_TEI_PATH_MAPPING
    )


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

    def test_should_preserve_all_sub_tags(self, annotator: MagicMock):
        structured_document = _structured_document_with_sub_elements(
            E.sub1('sub1'),
            E.sub2('sub2')
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == 'sub1'
        assert get_xpath_text(_get_root(structured_document), '//sub2') == 'sub2'

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=True,
            preserve_sub_tags=True,
            fields=['title']
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == 'sub1'
        assert get_xpath_text(_get_root(structured_document), '//sub2') == 'sub2'

    def test_should_preserve_some_sub_tags(self, annotator: MagicMock):
        structured_document = _structured_document_with_sub_elements(
            E.sub1('sub1'),
            E.sub2('sub2')
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == 'sub1'
        assert get_xpath_text(_get_root(structured_document), '//sub2') == 'sub2'

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=True,
            preserve_sub_tags=True,
            no_preserve_sub_fields={'sub1'},
            fields=['title']
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == ''
        assert get_xpath_text(_get_root(structured_document), '//sub2') == 'sub2'

    def test_should_not_preserve_sub_tags(self, annotator: MagicMock):
        structured_document = _structured_document_with_sub_elements(
            E.sub1('sub1'),
            E.sub2('sub2')
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == 'sub1'
        assert get_xpath_text(_get_root(structured_document), '//sub2') == 'sub2'

        annotate_structured_document_inplace(
            structured_document,
            annotator=annotator,
            preserve_tags=True,
            preserve_sub_tags=False,
            fields=['title']
        )
        assert get_xpath_text(_get_root(structured_document), '//sub1') == ''
        assert get_xpath_text(_get_root(structured_document), '//sub2') == ''
