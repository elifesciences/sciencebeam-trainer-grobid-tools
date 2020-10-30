import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.collection import flatten

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
CAPTION_PARAGRAPH_2 = 'Caption Paragraph 2'

LONG_DATA_TEXT_1 = ('Some data ' * 10).strip()
LONG_ATTRIB_TEXT_1 = 'Some long long attrib contents 1'


TEI_BY_JATS_REF_TYPE_MAP = {
    'bibr': 'biblio',
    'fig': 'figure',
    'table': 'table',
    'disp-formula': 'formula',
    'sec': 'section',
    'boxed-text': 'box'
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
                'section_title',
                'section_paragraph'
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
            'fields': 'section_title'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [
            SECTION_LABEL_1 + ' ' + SECTION_TITLE_1
        ]

    def test_should_auto_annotate_single_top_level_body_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # e.g. `153445v1` contains paragraphs as direct children of the body
        target_body_content_nodes = [E.p(TEXT_1)]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1]

    def test_should_auto_annotate_single_top_level_list_as_list_and_items(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # e.g. `214296v1` contains list as direct children of the body
        target_body_content_nodes = [E.list(
            E('list-item', TEXT_1),
            ' list-text ',
            E('list-item', TEXT_2)
        )]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraph',
                'list_item'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//list/item') == [
            TEXT_1, TEXT_2
        ]
        assert get_xpath_text_list(tei_auto_root, '//list') == [
            TEXT_1 + ' list-text ' + TEXT_2
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
                'section_title',
                'section_paragraph'
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
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1]

    def test_should_auto_annotate_single_back_ref_list_section_title(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E.sec(E.title(SECTION_TITLE_1)),
            ' ',
            E('ref-list', *[
                E.title('References'),
            ])
        ]
        tei_text = get_nodes_text(target_back_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//other') == ['References']

    def test_should_auto_annotate_multiple_section_title_and_paragraphs(
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
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1, TEXT_2]

    def test_should_auto_annotate_multiple_out_of_order_section_title_and_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        section_titles = ['First Title', 'Second Heading']
        section_paragraphs = [TEXT_1, TEXT_2]
        target_section_1_content_nodes = [
            E.sec(
                E.title(section_titles[0]),
                ' ',
                E.p(section_paragraphs[0]),
            )
        ]
        target_section_2_content_nodes = [
            E.sec(
                E.title(section_titles[1]),
                ' ',
                E.p(section_paragraphs[1]),
            )
        ]
        target_body_content_nodes = [
            *target_section_1_content_nodes,
            ' ',
            *target_section_2_content_nodes
        ]
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                get_nodes_text(target_section_2_content_nodes),
                *flatten([(E.lb(), ' x ') for _ in range(100)]),
                get_nodes_text(target_section_1_content_nodes),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher-lookahead-lines': 10,
            'fields': ','.join([
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == list(reversed(section_titles))
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_2, TEXT_1]

    def test_should_auto_annotate_nested_section_title_and_paragraphs(
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
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1, SECTION_TITLE_2]
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1, TEXT_2]

    @pytest.mark.skip('not yet implemented')
    def test_should_auto_annotate_single_paragraphs_split_by_figure(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.p(TEXT_1 + ' ' + TEXT_2)
            )
        ]
        tei_text = TEXT_1 + ' ' + LONG_DATA_TEXT_1 + ' ' + TEXT_2
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//p') == [TEXT_1 + ' ' + TEXT_2]

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
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        for key, tei_type_value in TEI_BY_JATS_REF_TYPE_MAP.items():
            assert get_xpath_text_list(
                tei_auto_root, '//p/ref[@type="%s"]' % tei_type_value
            ) == [CITATION_TEXT_BY_JATS_REF_TYPE_MAP[key]]
        assert get_xpath_text_list(tei_auto_root, '//p') == [paragraph_text]

    def test_should_auto_annotate_single_paragraph_citations_in_list_items_inside_paragraphs(
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
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        for key, tei_type_value in TEI_BY_JATS_REF_TYPE_MAP.items():
            assert get_xpath_text_list(
                tei_auto_root, '//p/ref[@type="%s"]' % tei_type_value
            ) == [CITATION_TEXT_BY_JATS_REF_TYPE_MAP[key]]
        assert get_xpath_text_list(tei_auto_root, '//p') == [paragraph_text]

    def test_should_auto_annotate_single_paragraph_citations_in_list_items_outside_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_paragraph_content_nodes = [TEXT_1, ' ']
        for key, value in CITATION_TEXT_BY_JATS_REF_TYPE_MAP.items():
            target_paragraph_content_nodes.append(E.xref({'ref-type': key}, value))
            target_paragraph_content_nodes.append(' ')
        target_paragraph_content_nodes.append(TEXT_2)
        target_body_content_nodes = [E.sec(
            E.list(E('list-item', *target_paragraph_content_nodes))
        ), ' Other']
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
                'section_paragraph',
                'list_item'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        for key, tei_type_value in TEI_BY_JATS_REF_TYPE_MAP.items():
            assert get_xpath_text_list(
                tei_auto_root, '//item/ref[@type="%s"]' % tei_type_value
            ) == [CITATION_TEXT_BY_JATS_REF_TYPE_MAP[key]]
        assert get_xpath_text_list(tei_auto_root, '//list/item') == [paragraph_text]

    def test_should_auto_annotate_single_paragraph_citations_in_boxed_text_inside_lists(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_paragraph_content_nodes = [TEXT_1, ' ']
        for key, value in CITATION_TEXT_BY_JATS_REF_TYPE_MAP.items():
            target_paragraph_content_nodes.append(E.xref({'ref-type': key}, value))
            target_paragraph_content_nodes.append(' ')
        target_paragraph_content_nodes.append(TEXT_2)
        target_body_content_nodes = [E.sec(E('boxed-text', *[
            E.label(LABEL_1),
            ' ',
            E.caption(SECTION_TITLE_1),
            ' ',
            E.p(
                E.list(E('list-item', *target_paragraph_content_nodes))
            )
        ])), ' Other']
        paragraph_text = get_nodes_text(target_paragraph_content_nodes)
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([E.other(tei_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_paragraph',
                'boxed_text_title',
                'boxed_text_paragraph'
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
        target_figure_label_caption_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
                ' ',
                E.p(CAPTION_PARAGRAPH_1)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E.fig(*target_figure_label_caption_content_nodes),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph',
                'figure',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//figure[not(@type="table")]') == [
            get_nodes_text(target_figure_label_caption_content_nodes)
        ]

    def test_should_auto_annotate_single_figure_label_description_with_attrib(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_label_caption_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
                ' ',
                E.p(CAPTION_PARAGRAPH_1)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E.fig(*target_figure_label_caption_content_nodes, *[
                    ' ',
                    E.attrib(LONG_ATTRIB_TEXT_1)
                ])
            )
        ]
        tei_text = (
            get_nodes_text(target_figure_label_caption_content_nodes)
            + LONG_DATA_TEXT_1
            + ' '
            + LONG_ATTRIB_TEXT_1
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'figure',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//figure[not(@type="table")]') == [
            tei_text
        ]

    def test_should_auto_annotate_single_figure_data_before_label_description(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_label_caption_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
                ' ',
                E.p(CAPTION_PARAGRAPH_1)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E.fig(*target_figure_label_caption_content_nodes)
            )
        ]
        tei_text = (
            LONG_DATA_TEXT_1
            + ' '
            + get_nodes_text(target_figure_label_caption_content_nodes)
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'figure',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//figure[not(@type="table")]') == [
            tei_text
        ]

    def test_should_auto_annotate_single_table_label_description(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_table_label_caption_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
                ' ',
                E.p(CAPTION_PARAGRAPH_1)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                ' ',
                E.p(TEXT_1),
                ' ',
                E('table-wrap', *target_table_label_caption_content_nodes),
                ' ',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph',
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
        assert get_xpath_text_list(tei_auto_root, '//figure[@type="table"]') == [
            get_nodes_text(target_table_label_caption_content_nodes)
        ]

    def test_should_auto_annotate_single_table_label_description_with_attrib(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_table_label_caption_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
                ' ',
                E.p(CAPTION_PARAGRAPH_1)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E('table-wrap', *target_table_label_caption_content_nodes, *[
                    ' ',
                    E.attrib(LONG_ATTRIB_TEXT_1)
                ]),
            )
        ]
        tei_text = (
            get_nodes_text(target_table_label_caption_content_nodes)
            + LONG_DATA_TEXT_1
            + ' '
            + LONG_ATTRIB_TEXT_1
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//figure[@type="table"]') == [
            tei_text
        ]

    def test_should_auto_annotate_single_boxed_text(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_boxed_text_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
            ),
            ' ',
            # in `306415v1` the paragraph is outside the caption
            E.p(CAPTION_PARAGRAPH_1)
        ]
        target_body_content_nodes = [
            E.sec(
                E('boxed-text', *target_boxed_text_content_nodes)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph',
                'boxed_text_title',
                'boxed_text_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head[@type="box"]') == [
            LABEL_1 + ' ' + CAPTION_TITLE_1
        ]
        assert get_xpath_text_list(tei_auto_root, '//p[@type="box"]') == [
            CAPTION_PARAGRAPH_1
        ]

    def test_should_ignore_nested_paragraphs_in_boxed_text(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_boxed_text_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(
                E.title(CAPTION_TITLE_1),
            ),
            ' ',
            # in `306415v1` the paragraph is outside the caption
            E.p(
                CAPTION_PARAGRAPH_1,
                ' ',
                E.p(CAPTION_PARAGRAPH_2)
            )
        ]
        target_body_content_nodes = [
            E.sec(
                E('boxed-text', *target_boxed_text_content_nodes)
            )
        ]
        tei_text = get_nodes_text(target_body_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([tei_text])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph',
                'boxed_text_title',
                'boxed_text_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//head[@type="box"]') == [
            LABEL_1 + ' ' + CAPTION_TITLE_1
        ]
        assert get_xpath_text_list(tei_auto_root, '//p[@type="box"]') == [
            CAPTION_PARAGRAPH_1 + ' ' + CAPTION_PARAGRAPH_2
        ]

    def test_should_annotate_references_title(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E('ref-list', E.title(SECTION_TITLE_1))
        ]
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                TEXT_1,
                ' ',
                SECTION_TITLE_1,
                ' ',
                TEXT_2
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph',
                'reference_list_title'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//other[@type="ref-list-title"]') == [
            SECTION_TITLE_1
        ]

    def test_should_convert_note_other_to_other(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = []
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.note({'type': 'other'}, TEXT_1),
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(body_nodes=target_body_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//other') == [TEXT_1]

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
                'section_title',
                'section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//formula') == [TEXT_1]
        assert get_xpath_text_list(tei_auto_root, '//head') == [SECTION_TITLE_1]
