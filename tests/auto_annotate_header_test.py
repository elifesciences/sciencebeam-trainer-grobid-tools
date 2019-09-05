import logging
from pathlib import Path

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.auto_annotate_header import (
    main
)

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import get_target_xml_node, get_xpath_text


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.header.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).header.tei.xml/\1.xml/'

TEXT_1 = 'text 1'


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


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_title(
            self, test_helper: SingleFileEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            E.tei(E.text(E.front(
                E.note(TEXT_1)
            )))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TEXT_1)
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TEXT_1
