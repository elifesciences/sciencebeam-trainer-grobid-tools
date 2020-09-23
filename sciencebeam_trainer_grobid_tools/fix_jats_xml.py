import argparse
import concurrent
import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Tuple, Optional

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


XLINK_NS = 'http://www.w3.org/1999/xlink'
XLINK_HREF = '{%s}href' % XLINK_NS

REF_XPATH = './/back/ref-list/ref'
MIXED_CITATION_XPATH = './/mixed-citation'
DOI_XPATH = './/pub-id[@pub-id-type="doi"]'
PII_XPATH = './/pub-id[@pub-id-type="pii"]'
PMID_XPATH = './/pub-id[@pub-id-type="pmid"]'
PMCID_XPATH = './/pub-id[@pub-id-type="pmcid"]'

DOI_PATTERN = r'\b(10.\d{4,}/[^\[]+)'
PII_PATTERN = r'\b(?:doi\:)?(\S{5,})\s*\[pii\]'
PMID_FIX_PATTERN = r'(?:PMID\s*\:\s*)?(\d{1,})'
PMID_PATTERN = r'(?:PMID\s*\:\s*)(\d{1,})'
PMCID_PATTERN = r'(PMC\d{7,})'

DOI_URL_PREFIX_PATTERN = r'((?:https?\s*\:\s*/\s*/\s*)?doi\s*.\s*org\s*/\s*)'


def with_element_tail(element: etree.Element, tail: str) -> etree.Element:
    element.tail = tail
    return element


def get_jats_pii_element(pii: str, tail: str = None) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pii'}, pii)
    if tail:
        node.tail = tail
    return node


def get_jats_pmid_element(pmid: str, tail: str = None) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pmid'}, pmid)
    if tail:
        node.tail = tail
    return node


def get_jats_pmcid_element(pmcid: str, tail: str = None) -> etree.Element:
    node = E('pub-id', {'pub-id-type': 'pmcid'}, pmcid)
    if tail:
        node.tail = tail
    return node


def get_full_cleaned_url(text: str):
    url = re.sub(r'\s', '', text)
    if '://' not in url:
        url = 'https://' + url
    return url


def get_jats_ext_link_element(
        text: str,
        tail: str = None,
        ext_link_type: str = 'uri',
        url: str = None) -> etree.Element:
    if url is None:
        url = get_full_cleaned_url(text)
    node = E(
        'ext-link',
        {
            'ext-link-type': ext_link_type,
            XLINK_HREF: url
        },
        text
    )
    if tail:
        node.tail = tail
    return node


def get_previous_text(current: etree.Element) -> str:
    previous = current.getprevious()
    if previous is not None:
        return previous.tail
    else:
        return current.getparent().text


def set_previous_text(current: etree.Element, text: str):
    previous = current.getprevious()
    if previous is not None:
        previous.tail = text
    else:
        parent = current.getparent()
        parent.text = text


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


def find_re_pattern_start_end(
        text: str,
        pattern: str,
        group_index: int = 1) -> Optional[Tuple[int, int]]:
    m = re.search(pattern, text)
    if not m:
        LOGGER.debug('pattern (%r) not found in: %r', pattern, text)
        return None
    return m.start(group_index), m.end(group_index)


def find_pii_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PII_PATTERN)


def find_pmid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PMID_PATTERN)


def find_pmcid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PMCID_PATTERN)


def add_annotation_to_element_text_if_matching(
        element: etree.Element,
        find_start_end_fn: Callable[[str], Optional[Tuple[int, int]]],
        create_element_fn: Callable[[str], etree.Element],
        as_next_sibling: bool = False) -> bool:
    text = element.text
    if not text:
        return False
    start_end = find_start_end_fn(text)
    if not start_end:
        LOGGER.debug('%s not found in: %r', find_start_end_fn, text)
        return False
    start, end = start_end
    matching_text = text[start:end]
    LOGGER.debug('matching: %s (%r)', start_end, matching_text)
    element.text = text[:start]
    new_element = with_element_tail(
        create_element_fn(matching_text),
        tail=text[end:]
    )
    if as_next_sibling:
        element.getparent().insert(
            element.getparent().index(element) + 1,
            new_element
        )
    else:
        element.insert(0, new_element)
    return True


def add_annotation_to_element_tail_if_matching(
        element: etree.Element,
        find_start_end_fn: Callable[[str], Optional[Tuple[int, int]]],
        create_element_fn: Callable[[str], etree.Element]) -> bool:
    text = element.tail
    if not text:
        return False
    start_end = find_start_end_fn(text)
    if not start_end:
        LOGGER.debug('%s not found in: %r', find_start_end_fn, text)
        return False
    start, end = start_end
    matching_text = text[start:end]
    LOGGER.debug('matching: %s (%r)', start_end, matching_text)
    element.getparent().insert(
        element.getparent().index(element) + 1,
        with_element_tail(
            create_element_fn(matching_text),
            tail=text[end:]
        )
    )
    element.tail = text[:start]
    return True


def add_annotation_to_element_if_matching(
        element: etree.Element,
        find_start_end_fn: Callable[[str], Optional[Tuple[int, int]]],
        create_element_fn: Callable[[str], etree.Element],
        parse_comment: bool) -> bool:
    if add_annotation_to_element_text_if_matching(
        element,
        find_start_end_fn=find_start_end_fn,
        create_element_fn=create_element_fn
    ):
        return True
    for child_element in element.xpath('./*'):
        if add_annotation_to_element_tail_if_matching(
            child_element,
            find_start_end_fn=find_start_end_fn,
            create_element_fn=create_element_fn
        ):
            return True
    if parse_comment:
        for child_element in element.xpath('./comment'):
            if add_annotation_to_element_text_if_matching(
                child_element,
                find_start_end_fn=find_start_end_fn,
                create_element_fn=create_element_fn,
                as_next_sibling=True
            ):
                break
    return False


def add_annotation_to_reference_element_if_matching(
        reference_element: etree.Element,
        *args, **kwargs) -> bool:
    for mixed_citation_element in reference_element.xpath(MIXED_CITATION_XPATH):
        if add_annotation_to_element_if_matching(
            mixed_citation_element,
            *args,
            **kwargs
        ):
            return True
    return False


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
    for doi_element in reference_element.xpath(DOI_XPATH):
        previous_text = get_previous_text(doi_element)
        m = re.search(DOI_URL_PREFIX_PATTERN, previous_text)
        if not m:
            LOGGER.debug('not matching doi url prefix: %r', previous_text)
            continue
        matching_doi_url_prefix = m.group(1)
        doi_url = matching_doi_url_prefix + doi_element.text
        LOGGER.debug('m: %s (%r)', m, matching_doi_url_prefix)
        set_previous_text(doi_element, previous_text[:m.start(1)])
        doi_element.getparent().replace(
            doi_element,
            get_jats_ext_link_element(
                doi_url,
                tail=doi_element.tail
            )
        )
    return reference_element


def fix_pmid(reference_element: etree.Element) -> etree.Element:
    for pmid_element in reference_element.xpath(PMID_XPATH):
        pmid_text = pmid_element.text
        m = re.search(PMID_FIX_PATTERN, pmid_text)
        if not m:
            LOGGER.debug('not matching pmid: %r', pmid_text)
            replace_element_with_text(pmid_element, pmid_text)
            continue
        matching_pmid = m.group(1).rstrip()
        LOGGER.debug('m: %s (%r)', m, matching_pmid)
        pmid_element.text = matching_pmid
        add_text_to_previous(pmid_element, pmid_text[:m.start(1)])
        add_text_to_tail_prefix(pmid_element, pmid_text[m.start(1) + len(matching_pmid):])
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
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pii_start_end,
        create_element_fn=get_jats_pii_element,
        parse_comment=False
    )
    return reference_element


def add_pmid_annotation_if_not_present(reference_element: etree.Element) -> etree.Element:
    if reference_element.xpath(PMID_XPATH):
        return reference_element
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pmid_start_end,
        create_element_fn=get_jats_pmid_element,
        parse_comment=True
    )
    return reference_element


def add_pmcid_annotation_if_not_present(reference_element: etree.Element) -> etree.Element:
    if reference_element.xpath(PMCID_XPATH):
        return reference_element
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pmcid_start_end,
        create_element_fn=get_jats_pmcid_element,
        parse_comment=True
    )
    return reference_element


def fix_reference(reference_element: etree.Element) -> etree.Element:
    fix_doi(reference_element)
    fix_pmid(reference_element)
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
