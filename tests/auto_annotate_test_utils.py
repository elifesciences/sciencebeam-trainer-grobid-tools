import logging
from pathlib import Path

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content

from .test_utils import dict_to_args


LOGGER = logging.getLogger(__name__)

TOKEN_1 = 'token1'

XML_FILENAME_1 = 'document1.xml'


class SingleFileAutoAnnotateEndToEndTestHelper:
    def __init__(
            self,
            temp_dir: Path,
            tei_filename: str,
            tei_filename_regex: str):
        self.tei_raw_path = temp_dir.joinpath('tei-raw')
        self.tei_auto_path = temp_dir.joinpath('tei-auto')
        self.xml_path = temp_dir.joinpath('xml')
        self.tei_raw_path.mkdir()
        self.xml_path.mkdir()
        self.tei_raw_file_path = self.tei_raw_path.joinpath(tei_filename)
        self.xml_file_path = self.xml_path.joinpath(XML_FILENAME_1)
        self.main_args_dict = {
            'source-base-path': self.tei_raw_path,
            'output-path': self.tei_auto_path,
            'xml-path': self.xml_path,
            'xml-filename-regex': tei_filename_regex,
            'fields': 'title,abstract'
        }
        self.main_args = dict_to_args(self.main_args_dict)
        self.tei_auto_file_path = self.tei_auto_path.joinpath(tei_filename)

    def get_tei_auto_root(self):
        assert self.tei_auto_file_path.exists()
        tei_auto_root = etree.parse(str(self.tei_auto_file_path)).getroot()
        LOGGER.info('tei_auto_root: %s', etree.tostring(tei_auto_root))
        return tei_auto_root


def get_default_tei_node() -> etree.Element:
    return E.tei(E.text(E.note(TOKEN_1)))


def get_target_xml_node(title: str = None) -> etree.Element:
    front_node = E.front()
    if title:
        front_node.append(E('article-meta', E('title-group', E('article-title', title))))
    return E.article(front_node)


def get_default_target_xml_node():
    return get_target_xml_node(title=TOKEN_1)


def get_xpath_text(root: etree.Element, xpath: str, delimiter: str = ' ') -> str:
    return delimiter.join(get_text_content(node) for node in root.xpath(xpath))
