from typing import List

from lxml import etree
from lxml.builder import ElementMaker


from sciencebeam_trainer_grobid_tools.utils.xml import (
    get_xpath_matches,
    get_first_xpath_match,
    get_xpath_text
)


TEI_NS = 'http://www.tei-c.org/ns/1.0'

TEI_NS_MAP = {
    'tei': TEI_NS
}

TEI_E = ElementMaker(namespace=TEI_NS, nsmap=TEI_NS_MAP)


def get_tei_xpath_matches(
        parent: etree.Element,
        xpath: str,
        **kwargs) -> List[etree.Element]:
    return get_xpath_matches(parent, xpath, namespaces=TEI_NS_MAP, **kwargs)


def get_first_tei_xpath_match(
        parent: etree.Element,
        xpath: str,
        **kwargs) -> etree.Element:
    return get_first_xpath_match(parent, xpath, namespaces=TEI_NS_MAP, **kwargs)


def get_tei_xpath_text(*args, **kwargs):
    return get_xpath_text(*args, namespaces=TEI_NS_MAP, **kwargs)
