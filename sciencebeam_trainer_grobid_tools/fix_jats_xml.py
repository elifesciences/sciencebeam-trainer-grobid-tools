import argparse
import concurrent
import logging
import os
import re
from pathlib import Path
from typing import List

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_utils.utils.file_path import (
    relative_path
)

from sciencebeam_utils.beam_utils.files import find_matching_filenames_with_limit

from sciencebeam_trainer_grobid_tools.utils.progress_logger import (
    logging_tqdm
)
from sciencebeam_trainer_grobid_tools.utils.xml import parse_xml

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument
)


LOGGER = logging.getLogger(__name__)


REF_XPATH = './/back/ref-list/ref'
MIXED_CITATION_XPATH = './/mixed-citation'
DOI_XPATH = './/pub-id[@pub-id-type="doi"]'
PII_XPATH = './/pub-id[@pub-id-type="pii"]'
PMID_XPATH = './/pub-id[@pub-id-type="pmid"]'
PMCID_XPATH = './/pub-id[@pub-id-type="pmcid"]'

DOI_PATTERN = r'\b(10.\d{4,}/[^\[]+)'
PII_PATTERN = r'\b(?:doi\:)?(\S{5,})\s*\[pii\]'
PMID_PATTERN = r'(?:PMID\s*\:\s*)(\d{1,})'
PMCID_PATTERN = r'(PMC\d{7,})'


def get_jats_pii_element(pii: str, tail: str) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pii'}, pii)
    if tail:
        node.tail = tail
    return node


def get_jats_pmid_element(pmid: str, tail: str) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pmid'}, pmid)
    if tail:
        node.tail = tail
    return node


def get_jats_pmcid_element(pmcid: str, tail: str) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pmcid'}, pmcid)
    if tail:
        node.tail = tail
    return node


def add_text_to_previous(current: etree.Element, text: str):
    previous = current.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or '') + text
    else:
        parent = current.getparent()
        parent.text = (parent.text or '') + text


def add_text_to_tail_prefix(current: etree.Element, text: str):
    current.tail = text + (current.tail or '')


def replace_element_with_text(current: etree.Element, text: str):
    add_text_to_previous(current, text)
    current.getparent().remove(current)


def fix_doi(reference_element: etree.Element) -> etree.Element:
    for doi_element in reference_element.xpath(DOI_XPATH):
        doi_text = doi_element.text
        m = re.search(DOI_PATTERN, doi_text)
        if not m:
            LOGGER.debug('not matching doi: %r', doi_text)
            replace_element_with_text(doi_element, doi_text)
            continue
        matching_doi = m.group(1).rstrip()
        LOGGER.debug('m: %s (%r)', m, matching_doi)
        doi_element.text = matching_doi
        add_text_to_previous(doi_element, doi_text[:m.start(1)])
        add_text_to_tail_prefix(doi_element, doi_text[m.start(1) + len(matching_doi):])
    return reference_element


def fix_pmcid(reference_element: etree.Element) -> etree.Element:
    for pmcid_element in reference_element.xpath(PMCID_XPATH):
        pmcid_text = pmcid_element.text
        m = re.search(PMCID_PATTERN, pmcid_text)
        if not m:
            LOGGER.debug('not matching pmcid: %r', pmcid_text)
            replace_element_with_text(pmcid_element, pmcid_text)
            continue
        matching_pmcid = m.group(1).rstrip()
        LOGGER.debug('m: %s (%r)', m, matching_pmcid)
        pmcid_element.text = matching_pmcid
        add_text_to_previous(pmcid_element, pmcid_text[:m.start(1)])
        add_text_to_tail_prefix(pmcid_element, pmcid_text[m.start(1) + len(matching_pmcid):])
    return reference_element


def add_pii_annotation_if_not_present(reference_element: etree.Element) -> etree.Element:
    if reference_element.xpath(PII_XPATH):
        return reference_element
    for mixed_citation_element in reference_element.xpath(MIXED_CITATION_XPATH):
        mixed_citation_text = mixed_citation_element.text
        if not mixed_citation_text:
            continue
        m = re.search(PII_PATTERN, mixed_citation_text)
        if not m:
            LOGGER.debug('pii not found in: %r', mixed_citation_text)
            continue
        matching_pii = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pii)
        mixed_citation_element.text = mixed_citation_text[:m.start(1)]
        mixed_citation_element.insert(0, get_jats_pii_element(
            matching_pii,
            tail=mixed_citation_text[m.end(1):]
        ))
    for child_element in reference_element.xpath(MIXED_CITATION_XPATH + '/*'):
        child_tail_text = child_element.tail
        if not child_tail_text:
            continue
        m = re.search(PII_PATTERN, child_tail_text)
        if not m:
            LOGGER.debug('pii not found in: %r', child_tail_text)
            continue
        matching_pii = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pii)
        child_element.getparent().insert(
            child_element.getparent().index(child_element) + 1,
            get_jats_pii_element(
                matching_pii,
                tail=child_tail_text[m.end(1):]
            )
        )
        child_element.tail = child_tail_text[:m.start(1)]
    return reference_element


def add_pmid_annotation_if_not_present(reference_element: etree.Element) -> etree.Element:
    if reference_element.xpath(PMID_XPATH):
        return reference_element
    for mixed_citation_element in reference_element.xpath(MIXED_CITATION_XPATH):
        mixed_citation_text = mixed_citation_element.text
        if not mixed_citation_text:
            continue
        m = re.search(PMID_PATTERN, mixed_citation_text)
        if not m:
            LOGGER.debug('pmid not found in: %r', mixed_citation_text)
            continue
        matching_pii = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pii)
        mixed_citation_element.text = mixed_citation_text[:m.start(1)]
        mixed_citation_element.insert(0, get_jats_pmid_element(
            matching_pii,
            tail=mixed_citation_text[m.end(1):]
        ))
    for child_element in reference_element.xpath(MIXED_CITATION_XPATH + '/*'):
        child_tail_text = child_element.tail
        if not child_tail_text:
            continue
        m = re.search(PMID_PATTERN, child_tail_text)
        if not m:
            LOGGER.debug('pmid not found in: %r', child_tail_text)
            continue
        matching_pii = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pii)
        child_element.getparent().insert(
            child_element.getparent().index(child_element) + 1,
            get_jats_pmid_element(
                matching_pii,
                tail=child_tail_text[m.end(1):]
            )
        )
        child_element.tail = child_tail_text[:m.start(1)]
    return reference_element


def add_pmcid_annotation_if_not_present(reference_element: etree.Element) -> etree.Element:
    if reference_element.xpath(PMCID_XPATH):
        return reference_element
    for mixed_citation_element in reference_element.xpath(MIXED_CITATION_XPATH):
        mixed_citation_text = mixed_citation_element.text
        if not mixed_citation_text:
            continue
        m = re.search(PMCID_PATTERN, mixed_citation_text)
        if not m:
            LOGGER.debug('pmcid not found in: %r', mixed_citation_text)
            continue
        matching_pmcid = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pmcid)
        mixed_citation_element.text = mixed_citation_text[:m.start(1)]
        mixed_citation_element.insert(0, get_jats_pmcid_element(
            matching_pmcid,
            tail=mixed_citation_text[m.end(1):]
        ))
    for child_element in reference_element.xpath(MIXED_CITATION_XPATH + '/*'):
        child_tail_text = child_element.tail
        if not child_tail_text:
            continue
        m = re.search(PMCID_PATTERN, child_tail_text)
        if not m:
            LOGGER.debug('pmcid not found in: %r', child_tail_text)
            continue
        matching_pmcid = m.group(1)
        LOGGER.debug('m: %s (%r)', m, matching_pmcid)
        child_element.getparent().insert(
            child_element.getparent().index(child_element) + 1,
            get_jats_pmcid_element(
                matching_pmcid,
                tail=child_tail_text[m.end(1):]
            )
        )
        child_element.tail = child_tail_text[:m.start(1)]
    return reference_element


def fix_reference(reference_element: etree.Element) -> etree.Element:
    fix_doi(reference_element)
    fix_pmcid(reference_element)
    add_pmid_annotation_if_not_present(reference_element)
    add_pmcid_annotation_if_not_present(reference_element)
    add_pii_annotation_if_not_present(reference_element)
    return reference_element


def fix_jats_xml_node(root: etree.Element):
    for ref in root.xpath(REF_XPATH):
        fix_reference(ref)
    return root


def fix_jats_xml_file(input_file: str, output_file: str):
    LOGGER.info('processing: %r -> %r', input_file, output_file)
    root = parse_xml(input_file)
    fix_jats_xml_node(root)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_bytes(etree.tostring(root))


class FixJatsProcessor:
    def __init__(self, opt: argparse.Namespace):
        self.num_workers = opt.num_workers
        self.multi_processing = opt.multi_processing
        self.source_path = opt.source_path
        self.source_base_path = opt.source_base_path or os.path.dirname(self.source_path)
        self.output_path = opt.output_path
        self.source_filename_pattern = opt.source_filename_pattern
        self.limit = opt.limit

    def get_output_file_for_source_file(self, source_url: str):
        return os.path.join(
            self.output_path,
            relative_path(self.source_base_path, source_url)
        )

    def process_source_file(self, source_file: str):
        output_file = self.get_output_file_for_source_file(source_file)
        assert output_file != source_file
        fix_jats_xml_file(source_file, output_file)

    def run_local_pipeline(self, xml_file_list: List[str]):
        num_workers = min(self.num_workers, len(xml_file_list))
        multi_processing = self.multi_processing
        LOGGER.info('using %d workers (multi_processing: %s)', num_workers, multi_processing)
        PoolExecutor = (
            concurrent.futures.ProcessPoolExecutor if multi_processing
            else concurrent.futures.ThreadPoolExecutor
        )
        with PoolExecutor(max_workers=num_workers) as executor:
            with logging_tqdm(total=len(xml_file_list)) as pbar:
                future_to_url = {
                    executor.submit(self.process_source_file, url): url
                    for url in xml_file_list
                }
                LOGGER.debug('future_to_url: %s', future_to_url)
                for future in concurrent.futures.as_completed(future_to_url):
                    pbar.update(1)
                    future.result()

    def get_source_file_list(self):
        if self.source_path:
            return [self.source_path]
        return list(find_matching_filenames_with_limit(os.path.join(
            self.source_base_path,
            self.source_filename_pattern
        ), limit=self.limit))

    def run(self):
        xml_file_list = self.get_source_file_list()
        if not xml_file_list:
            LOGGER.warning('no files found to process')
            return
        self.run_local_pipeline(xml_file_list)


def add_args(parser: argparse.ArgumentParser):
    source_group = parser.add_argument_group('source')
    source_group.add_argument(
        '--source-base-path', type=str,
        help='source base data path for files to fix'
    )
    source_group.add_argument(
        '--source-path', type=str,
        help='source path to a specific file to fix'
    )
    source_group.add_argument(
        '--source-filename-pattern', type=str,
        default='**.xml*',
        help='file pattern within source base path to find files to process'
    )

    parser.add_argument(
        '--output-path', type=str, required=True,
        help='output base path'
    )

    parser.add_argument(
        '--limit', type=int, required=False,
        help='limit the number of files to process'
    )

    parser.add_argument(
        '--multi-processing', action='store_true', default=False,
        help='enable multi processing rather than multi threading'
    )

    add_cloud_args(parser)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_args(parser)
    add_debug_argument(parser)

    parsed_args = parser.parse_args(argv)
    process_cloud_args(
        parsed_args,
        parsed_args.output_path,
        name='sciencebeam-grobid-trainer-tools'
    )
    LOGGER.info('parsed_args: %s', parsed_args)
    return parsed_args


def run(args: argparse.Namespace):
    FixJatsProcessor(args).run()


def main(argv=None):
    args = parse_args(argv)
    process_debug_argument(args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
