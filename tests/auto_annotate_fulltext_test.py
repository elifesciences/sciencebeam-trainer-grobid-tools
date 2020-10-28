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

SECTION_LABEL_1 = '1.1'

SECTION_TITLE_1 = 'Section Title 1'
SECTION_TITLE_2 = 'Section Title 2'

LABEL_1 = 'Label 1'
CAPTION_TITLE_1 = 'Caption Title 1'
CAPTION_PARAGRAPH_1 = 'Caption Paragraph 1'

TEI_BY_JATS_REF_TYPE_MAP = {
    'bibr': 'biblio',
    'fig': 'figure',
    'table': 'table',
    'disp-formula': 'formula',
    'sec': 'section'
}


CITATION_TEXT_BY_JATS_REF_TYPE_MAP = {
    key: 'Citation %d' % (1 + index)
    for index, key in enumerate(TEI_BY_JATS_REF_TYPE_MAP.keys())
}


def get_training_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(*items))


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_single_section_title_and_paragraphs(
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
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1]

    def test_should_auto_annotate_single_section_title_with_label(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.label(SECTION_LABEL_1),
                ' ',
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': 'section_titles'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [
            SECTION_LABEL_1 + ' ' + SECTION_TITLE_1
        ]

    def test_should_auto_annotate_single_back_section_title_and_paragraph(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1)
            )
        ]
        tei_text = get_nodes_text(target_back_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1]

    def test_should_auto_annotate_single_back_ack_section_title_and_paragraph(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E.ack(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1)
            )
        ]
        tei_text = get_nodes_text(target_back_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1]

    def test_should_auto_annotate_multiple_section_titles_and_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
            ),
            ' ',
            E.sec(
                E.title(SECTION_TITLE_2),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1, TEXT_2]

    def test_should_auto_annotate_nested_section_titles_and_paragraphs(
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
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1, TEXT_2]

    def test_should_auto_annotate_single_paragraph_citations(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_paragraph_content_nodes = [TEXT_1, ' ']
        for key, value in CITATION_TEXT_BY_JATS_REF_TYPE_MAP.items():
            target_paragraph_content_nodes.append(E.xref({'ref-type': key}, value))
            target_paragraph_content_nodes.append(' ')
        target_paragraph_content_nodes.append(TEXT_2)
        target_body_content_nodes = [E.sec(E.p(*target_paragraph_content_nodes)), ' Other']
        paragraph_text = get_nodes_text(target_paragraph_content_nodes)
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        for key, tei_type_value in TEI_BY_JATS_REF_TYPE_MAP.items():
            assert get_xpath_text_list(
                tei_auto_root, '//p/ref[@type="%s"]' % tei_type_value
            ) == [CITATION_TEXT_BY_JATS_REF_TYPE_MAP[key]]
        assert get_xpath_text_list(tei_auto_root, '//p') == [paragraph_text]


    def test_should_auto_annotate_single_paragraph_citations_in_list_items(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_paragraph_content_nodes = [TEXT_1, ' ']
        for key, value in CITATION_TEXT_BY_JATS_REF_TYPE_MAP.items():
            target_paragraph_content_nodes.append(E.xref({'ref-type': key}, value))
            target_paragraph_content_nodes.append(' ')
        target_paragraph_content_nodes.append(TEXT_2)
        target_body_content_nodes = [E.sec(E.p(
            E.list(E('list-item', *target_paragraph_content_nodes))
        )), ' Other']
        paragraph_text = get_nodes_text(target_paragraph_content_nodes)
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        for key, tei_type_value in TEI_BY_JATS_REF_TYPE_MAP.items():
            assert get_xpath_text_list(
                tei_auto_root, '//p/ref[@type="%s"]' % tei_type_value
            ) == [CITATION_TEXT_BY_JATS_REF_TYPE_MAP[key]]
        assert get_xpath_text_list(tei_auto_root, '//p') == [paragraph_text]

    def test_should_auto_annotate_single_figure_label_description(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E.fig(
                    E.label(LABEL_1),
                    ' ',
                    E.caption(
                        E.title(CAPTION_TITLE_1),
                        ' ',
                        E.p(CAPTION_PARAGRAPH_1)
                    )
                ),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'figure',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//figure[not(@type="table")]') == [
            ' '.join([LABEL_1, CAPTION_TITLE_1, CAPTION_PARAGRAPH_1])
        ]

    def test_should_auto_annotate_single_table_label_description(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E('table-wrap', *[
                    E.label(LABEL_1),
                    ' ',
                    E.caption(
                        E.title(CAPTION_TITLE_1),
                        ' ',
                        E.p(CAPTION_PARAGRAPH_1)
                    )
                ]),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.note(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//figure[@type="table"]') == [
            ' '.join([LABEL_1, CAPTION_TITLE_1, CAPTION_PARAGRAPH_1])
        ]

    def test_should_preserve_formula(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ' + TEXT_1
            )
        ]
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.note(SECTION_TITLE_1),
                ' ',
                E.formula(TEXT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_titles',
                'section_paragraphs'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//formula') == [TEXT_1]
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
