from typing import List

from lxml import etree


TEI_NS = 'http://www.tei-c.org/ns/1.0'

TEI_NS_MAP = {
    'tei': TEI_NS
}


def tei_xpath(parent: etree.Element, xpath: str) -> List[etree.Element]:
    return parent.xpath(xpath, namespaces=TEI_NS_MAP)
