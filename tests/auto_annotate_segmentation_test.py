import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.auto_annotate_segmentation import (
    main
)

from .test_utils import log_on_exception, dict_to_args

from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_default_target_xml_node,
    get_xpath_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.segmentation.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).segmentation.tei.xml/\1.xml/'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'


def get_segmentation_tei_node(
        text_items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(*text_items))


def get_default_tei_node() -> etree.Element:
    return get_segmentation_tei_node([E.note(TOKEN_1)])


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_title_as_front(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.note(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//text/front') == TOKEN_1

    @log_on_exception
    def test_should_auto_annotate_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.note(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//text/front') == TOKEN_1

    @log_on_exception
    def test_should_process_specific_file(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'source-base-path': None,
            'source-path': str(test_helper.tei_raw_file_path)
        }), save_main_session=False)

        assert test_helper.get_tei_auto_root() is not None

    @log_on_exception
    def test_should_skip_existing_output_file_if_resume_is_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        test_helper.tei_auto_file_path.parent.mkdir()
        test_helper.tei_auto_file_path.write_bytes(b'existing')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'resume': True
        }), save_main_session=False)

        assert test_helper.tei_auto_file_path.read_bytes() == b'existing'

    @log_on_exception
    def test_should_run_locally_without_beam_if_workers_more_than_one(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        test_helper.tei_auto_file_path.parent.mkdir()
        test_helper.tei_auto_file_path.write_bytes(b'existing')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'num_workers': 2
        }), save_main_session=False)
        assert test_helper.get_tei_auto_root() is not None

    @log_on_exception
    def test_should_write_debug_match(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper, temp_dir: Path):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        debug_match_path = temp_dir.joinpath('debug.csv')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'debug-match': str(debug_match_path)
        }), save_main_session=False)

        assert debug_match_path.exists()

    @log_on_exception
    def test_should_preserve_existing_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.page(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            E.article(E.front(
            ))
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//text/page') == TOKEN_1

    @log_on_exception
    def test_should_always_preserve_specified_existing_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TOKEN_1),
                E.lb(),
                E.page(TOKEN_2),
                E.lb(),
                E.note(TOKEN_3),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=' '.join([TOKEN_1, TOKEN_2, TOKEN_3]))
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'always-preserve-fields': 'page'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//text/page') == TOKEN_2

    @log_on_exception
    def test_should_not_preserve_exclude_existing_tag_and_use_body_by_default(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.page(TOKEN_1)])
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
        assert get_xpath_text(tei_auto_root, '//text/page') == ''
        assert get_xpath_text(tei_auto_root, '//text/body') == TOKEN_1
