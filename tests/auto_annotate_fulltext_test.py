import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.utils.xml import get_xpath_text_list

from sciencebeam_trainer_grobid_tools.auto_annotate_fulltext import (
    main
)

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_nodes_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.fulltext.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).fulltext.tei.xml/\1.xml/'

TEXT_1 = 'text 1'
TEXT_2 = 'text 1'

SECTION_TITLE_1 = 'Section Title 1'
SECTION_TITLE_2 = 'Section Title 2'


def get_header_tei_node(
        front_items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(E.front(*front_items)))


def get_default_tei_node() -> etree.Element:
    return get_header_tei_node([E.note(TEXT_1)])


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_single_section_title(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=[
                E.sec(
                    E.title(TEXT_1)
                )
            ])
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': 'body_section_titles'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [TEXT_1]

    def test_should_auto_annotate_multiple_section_titles(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E.title(SECTION_TITLE_2),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': 'body_section_titles'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]

    def test_should_auto_annotate_nested_section_titles(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E.sec(
                    E.title(SECTION_TITLE_2),
                    ' ',
                    E.p(TEXT_2)
                )
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': 'body_section_titles'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]
