import logging
from pathlib import Path

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content

from sciencebeam_trainer_grobid_tools.auto_annotate_segmentation import (
    main
)

from .test_utils import log_on_exception, dict_to_args


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.segmentation.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).segmentation.tei.xml/\1.xml/'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'


class SingleFileEndToEndTestHelper:
    def __init__(
            self,
            temp_dir: Path):
        self.tei_raw_path = temp_dir.joinpath('tei-raw')
        self.tei_auto_path = temp_dir.joinpath('tei-auto')
        self.xml_path = temp_dir.joinpath('xml')
        self.tei_raw_path.mkdir()
        self.xml_path.mkdir()
        self.tei_raw_file_path = self.tei_raw_path.joinpath(TEI_FILENAME_1)
        self.xml_file_path = self.xml_path.joinpath(XML_FILENAME_1)
        self.main_args_dict = {
            'source-base-path': self.tei_raw_path,
            'output-path': self.tei_auto_path,
            'xml-path': self.xml_path,
            'xml-filename-regex': TEI_FILENAME_REGEX,
            'fields': 'title,abstract'
        }
        self.main_args = dict_to_args(self.main_args_dict)
        self.tei_auto_file_path = self.tei_auto_path.joinpath(TEI_FILENAME_1)

    def get_tei_auto_root(self):
        assert self.tei_auto_file_path.exists()
        tei_auto_root = etree.parse(str(self.tei_auto_file_path)).getroot()
        LOGGER.info('tei_auto_root: %s', etree.tostring(tei_auto_root))
        return tei_auto_root


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileEndToEndTestHelper:
    return SingleFileEndToEndTestHelper(temp_dir)


def _get_default_tei_node():
    return E.tei(E.text(E.note(TOKEN_1)))


def _get_xml_node(title: str = None):
    front_node = E.front()
    if title:
        front_node.append(E('article-meta', E('title-group', E('article-title', title))))
    return E.article(front_node)


def _get_default_xml_node():
    return _get_xml_node(title=TOKEN_1)


def _get_xpath_text(root: etree.Element, xpath: str, delimiter: str = ' ') -> str:
    return delimiter.join(get_text_content(node) for node in root.xpath(xpath))


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_title_as_front(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(
                E.note(TOKEN_1)
            ))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            _get_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert _get_xpath_text(tei_auto_root, '//text/front') == TOKEN_1

    @log_on_exception
    def test_should_auto_annotate_using_simple_matcher(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(
                E.note(TOKEN_1)
            ))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            _get_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert _get_xpath_text(tei_auto_root, '//text/front') == TOKEN_1

    @log_on_exception
    def test_should_process_specific_file(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(_get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(_get_default_xml_node()))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'source-base-path': None,
            'source-path': str(test_helper.tei_raw_file_path)
        }), save_main_session=False)

        assert test_helper.get_tei_auto_root() is not None

    @log_on_exception
    def test_should_write_debug_match(
            self, test_helper: SingleFileEndToEndTestHelper, temp_dir: Path):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(_get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(_get_default_xml_node()))
        debug_match_path = temp_dir.joinpath('debug.csv')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'debug-match': str(debug_match_path)
        }), save_main_session=False)

        assert debug_match_path.exists()

    @log_on_exception
    def test_should_preserve_existing_tag(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(
                E.page(TOKEN_1)
            ))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            E.article(E.front(
            ))
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert _get_xpath_text(tei_auto_root, '//text/page') == TOKEN_1

    @log_on_exception
    def test_should_always_preserve_specified_existing_tag(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(
                E.note(TOKEN_1),
                E.lb(),
                E.page(TOKEN_2),
                E.lb(),
                E.note(TOKEN_3),
                E.lb()
            ))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            _get_xml_node(title=' '.join([TOKEN_1, TOKEN_2, TOKEN_3]))
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'always-preserve-fields': 'page'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert _get_xpath_text(tei_auto_root, '//text/page') == TOKEN_2

    @log_on_exception
    def test_should_not_preserve_exclude_existing_tag_and_use_body_by_default(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(
                E.page(TOKEN_1)
            ))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            E.article(E.front(
            ))
        ))
        main([
            *test_helper.main_args,
            '--no-preserve-fields=page'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert _get_xpath_text(tei_auto_root, '//text/page') == ''
        assert _get_xpath_text(tei_auto_root, '//text/body') == TOKEN_1
