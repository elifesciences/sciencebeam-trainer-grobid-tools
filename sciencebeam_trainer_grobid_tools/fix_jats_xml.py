import argparse
import logging
import re
from pathlib import Path

from lxml import etree

from sciencebeam_trainer_grobid_tools.utils.xml import parse_xml

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument
)


LOGGER = logging.getLogger(__name__)


REF_XPATH = './/back/ref-list/ref'
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


def fix_jats_xml_node(root: etree.Element):
    for ref in root.xpath(REF_XPATH):
        fix_doi(ref)
    return root


def fix_jats_xml_file(input_file: str, output_file: str):
    LOGGER.info('processing: %r -> %r', input_file, output_file)
    root = parse_xml(input_file)
    fix_jats_xml_node(root)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_bytes(etree.tostring(root))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--input', type=str, required=True,
        help='path to input xml file'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='path to output xml file'
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_args(parser)
    add_debug_argument(parser)

    parsed_args = parser.parse_args(argv)
    LOGGER.info('parsed_args: %s', parsed_args)
    return parsed_args


def run(args: argparse.Namespace):
    fix_jats_xml_file(
        args.input,
        args.output
    )


def main(argv=None):
    args = parse_args(argv)
    process_debug_argument(args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
