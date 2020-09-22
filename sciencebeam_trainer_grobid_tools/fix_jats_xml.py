import logging
import re

from lxml import etree


LOGGER = logging.getLogger(__name__)


DOI_XPATH = './/pub-id[@pub-id-type="doi"]'

DOI_PATTERN = r'\b(10.\d{4,}/[^\[]+)'


def add_text_to_previous(current: etree.Element, text: str):
    previous = current.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or '') + text
    else:
        parent = current.getparent()
        parent.text = (parent.text or '') + text


def add_text_to_tail_prefix(current: etree.Element, text: str):
    current.tail = text + (current.tail or '')


def fix_doi(reference_element: etree.Element) -> etree.Element:
    for doi_element in reference_element.xpath(DOI_XPATH):
        doi_text = doi_element.text
        m = re.search(DOI_PATTERN, doi_text)
        if not m:
            LOGGER.debug('not matching doi: %r', doi_text)
            continue
        matching_doi = m.group(1).rstrip()
        LOGGER.debug('m: %s (%r)', m, matching_doi)
        doi_element.text = matching_doi
        add_text_to_previous(doi_element, doi_text[:m.start(1)])
        add_text_to_tail_prefix(doi_element, doi_text[m.start(1) + len(matching_doi):])
    return reference_element
