import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content_list

from sciencebeam_trainer_grobid_tools.utils.tei_xml import (
    get_tei_xpath_matches,
    get_tei_xpath_text_list,
    get_tei_xpath_text
)

from sciencebeam_trainer_grobid_tools.auto_annotate_figure import main

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_nodes_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.figure.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).figure.tei.xml/\1.xml/'

FIGURE_XPATH = './text/figure'

TEXT_1 = 'text 1'
TEXT_2 = 'text 2'

LABEL_1 = '1'
LABEL_2 = '2'


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


def get_all_figures(root: etree.Element, **kwargs) -> List[etree.Element]:
    return get_tei_xpath_matches(root, FIGURE_XPATH, **kwargs)


def get_first_figure(root: etree.Element) -> etree.Element:
    return get_all_figures(root, required=True)[0]


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_single_figure_with_label_and_caption(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_content_nodes = [
            E.label(LABEL_1),
            ' ',
            E.caption(E.p(TEXT_1))
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(body_nodes=[
                E.fig(*target_figure_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.figure(get_nodes_text(target_figure_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'figure'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_figure = get_first_figure(tei_auto_root)
        assert get_tei_xpath_text(first_figure, './/label') == (
            LABEL_1
        )
        assert get_tei_xpath_text(first_figure, './figDesc') == (
            TEXT_1
        )

    def test_should_auto_annotate_multiple_figures(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_content_nodes_1 = [
            E.label(LABEL_1),
            ' ',
            E.caption(E.p(TEXT_1))
        ]
        target_figure_content_nodes_2 = [
            E.label(LABEL_2),
            ' ',
            E.caption(E.p(TEXT_2))
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(body_nodes=[
                E.fig(*target_figure_content_nodes_1),
                E.fig(*target_figure_content_nodes_2)
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.figure(get_nodes_text(target_figure_content_nodes_1)),
                E.figure(get_nodes_text(target_figure_content_nodes_2))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'figure'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_tei_xpath_text_list(tei_auto_root, './/label') == [
            LABEL_1, LABEL_2
        ]
        assert get_tei_xpath_text_list(tei_auto_root, './/figDesc') == [
            TEXT_1, TEXT_2
        ]

    def test_should_not_segment_figures_if_disabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_content_nodes_1 = [
            E.label(LABEL_1),
            ' ',
            E.caption(E.p(TEXT_1))
        ]
        target_figure_content_nodes_2 = [
            E.label(LABEL_2),
            ' ',
            E.caption(E.p(TEXT_2))
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(body_nodes=[
                E.fig(*target_figure_content_nodes_1),
                E.fig(*target_figure_content_nodes_2)
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.figure(get_nodes_text(
                    target_figure_content_nodes_1
                    + [' ']
                    + target_figure_content_nodes_2
                ))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'figure'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_text_content_list(get_all_figures(tei_auto_root)) == [
            get_nodes_text(target_figure_content_nodes_1)
            + ' ' + get_nodes_text(target_figure_content_nodes_2)
        ]

    def test_should_segment_figures_if_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_figure_content_nodes_1 = [
            E.label(LABEL_1),
            ' ',
            E.caption(E.p(TEXT_1))
        ]
        target_figure_content_nodes_2 = [
            E.label(LABEL_2),
            ' ',
            E.caption(E.p(TEXT_2))
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(body_nodes=[
                E.fig(*target_figure_content_nodes_1),
                E.fig(*target_figure_content_nodes_2)
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node([
                E.figure(get_nodes_text(
                    target_figure_content_nodes_1
                    + [' ']
                    + target_figure_content_nodes_2
                ))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'segment-figures': True,
            'fields': 'figure'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_text_content_list(get_all_figures(tei_auto_root)) == [
            get_nodes_text(target_figure_content_nodes_1),
            get_nodes_text(target_figure_content_nodes_2)
        ]
