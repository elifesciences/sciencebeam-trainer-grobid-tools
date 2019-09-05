from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content


TOKEN_1 = 'token1'


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
