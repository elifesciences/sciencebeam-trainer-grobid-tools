import logging
from pathlib import Path
from typing import List, Union

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


def _add_all(parent: etree.Element, children: List[etree.Element]):
    if not children:
        return
    for child in children:
        parent.append(child)


def get_target_xml_node(
        title: str = None,
        author_nodes: List[etree.Element] = None,
        affiliation_nodes: List[etree.Element] = None,
        abstract_node: etree.Element = None,
        body_nodes: List[etree.Element] = None,
        back_nodes: List[etree.Element] = None,
        reference_nodes: List[etree.Element] = None) -> etree.Element:
    contrib_group = E('contrib-group')
    article_meta_node = E('article-meta', contrib_group)
    front_node = E.front(article_meta_node)
    body_node = E.body()
    back_node = E.back()
    if title:
        article_meta_node.append(E('title-group', E('article-title', title)))
    _add_all(contrib_group, author_nodes)
    _add_all(contrib_group, affiliation_nodes)
    _add_all(body_node, body_nodes)
    _add_all(back_node, back_nodes)
    if abstract_node is not None:
        article_meta_node.append(abstract_node)
    if reference_nodes:
        back_node.append(E('ref-list', *reference_nodes))
    return E.article(front_node, body_node, back_node)


def get_default_target_xml_node():
    return get_target_xml_node(title=TOKEN_1)


def get_nodes_text(nodes: List[Union[str, etree.Element]]) -> str:
    return ''.join([
        str(node)
        if isinstance(node, str)
        else get_text_content(node)
        for node in nodes
    ])
