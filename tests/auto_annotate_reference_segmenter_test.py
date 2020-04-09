import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.auto_annotate_reference_segmenter import (
    main
)

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_xpath_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.references.referenceSegmenter.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).references.referenceSegmenter.tei.xml/\1.xml/'

TEXT_1 = 'text 1'

LABEL_1 = '1'
REFERENCE_TEXT_1 = 'reference A'

# LABEL_2 = '2'
# REFERENCE_TEXT_1 = 'reference B'


def get_reference_segmenter_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(*items))


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


def get_jats_reference_node(label: str, text: str) -> etree.Element:
    ref = E.ref()
    if label:
        ref.append(E.label(label))
    ref.append(E('mixed-citation', text))
    return ref


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_single_reference(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_segmenter_tei_node([
                E.note(LABEL_1 + ' ' + REFERENCE_TEXT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, REFERENCE_TEXT_1),
            ])
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//listBibl/bibl') == ' '.join([
            LABEL_1, REFERENCE_TEXT_1
        ])

    @log_on_exception
    def test_should_auto_annotate_label_within_reference(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_segmenter_tei_node([
                E.note(LABEL_1 + ' ' + REFERENCE_TEXT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, REFERENCE_TEXT_1),
            ])
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference',
            'xml-mapping-overrides': 'reference.use-raw-text=true'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//listBibl/bibl') == ' '.join([
            LABEL_1, REFERENCE_TEXT_1
        ])
        assert get_xpath_text(tei_auto_root, '//listBibl/bibl/label') == LABEL_1
