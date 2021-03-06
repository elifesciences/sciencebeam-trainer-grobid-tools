import argparse
import concurrent
import logging
import os
import re
from collections import Counter, OrderedDict
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Tuple, Optional

import regex

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)
from sciencebeam_utils.beam_utils.io import save_file_content

from sciencebeam_utils.utils.file_path import (
    relative_path
)

from sciencebeam_utils.utils.xml import (
    get_text_content
)
from sciencebeam_utils.utils.exceptions import get_serializable_exception

from sciencebeam_utils.beam_utils.files import find_matching_filenames_with_limit
from sciencebeam_utils.utils.file_list import load_file_list
from sciencebeam_utils.utils.progress_logger import logging_tqdm

from sciencebeam_trainer_grobid_tools.utils.xml import parse_xml
from sciencebeam_trainer_grobid_tools.utils.io import auto_download_input_file

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument
)


LOGGER = logging.getLogger(__name__)


XLINK_NS = 'http://www.w3.org/1999/xlink'
XLINK_HREF = '{%s}href' % XLINK_NS


class JatsXpaths:
    REF = './/back/ref-list/ref'
    MIXED_CITATION = './/mixed-citation'
    ARTICLE_TITLE = './/article-title'
    EXT_LINK = './/ext-link'
    PUB_ID = './/pub-id'
    DOI = './/pub-id[@pub-id-type="doi"]'
    PII = './/pub-id[@pub-id-type="pii"]'
    PMID = './/pub-id[@pub-id-type="pmid"]'
    PMCID = './/pub-id[@pub-id-type="pmcid"]'
    OTHER_PUB_ID = './/pub-id[@pub-id-type="other"]'


class SpecialChars:
    LSQUO = '\u2018'
    RSQUO = '\u2019'
    LDQUO = '\u201C'
    RDQUO = '\u201D'


LEFT_QUOTE_CHARS = {'"', SpecialChars.LSQUO, SpecialChars.LDQUO}

RIGHT_BY_LEFT_QUOTE_CHAR = {
    '"': '"',
    SpecialChars.LSQUO: SpecialChars.RSQUO,
    SpecialChars.LDQUO: SpecialChars.RSQUO
}


# https://en.wikipedia.org/wiki/Digital_Object_Identifier
DOI_PATTERN = r'\b(10\.\d{4,}(?:\.\d{1,})*/.+)'

# https://en.wikipedia.org/wiki/Publisher_Item_Identifier
PII_VALID_PATTERN = r'\b([S,B]\W*(?:[0-9xX]\W*){15,}[0-9xX])'
PII_OTHER_PATTERN = r'\b(?:doi\:)?(\S{5,})\s*\[pii\]'

PMID_FIX_PATTERN = r'(?:PMID\s*\:\s*)?\b(\d{1,10})\b'
PMID_PATTERN = r'(?:PMID\s*\:\s*)(\d{1,10})\b'
PMCID_PATTERN = r'(PMC\d{1,})'
WOS_PATTERN = r'(?:WOS\s*\:\s*)(\d{15,15})\b'

DOI_URL_PREFIX_PATTERN = r'((?:https?\s*\:\s*/\s*/\s*)?(?:[a-z]+\s*\.\s*)?doi\s*.\s*org\s*/\s*)'

ARTICLE_TITLE_PATTERN = r'^(.*?)(\;\s*PMC\d+|\s*,\s*)?$'


DOI_TRUNCATE_AT_TOKENS = {'PubMed', 'PMID', 'PMCID', 'Error', 'Epub', 'Accessed'}
DOI_TRUNCATE_AT_PATTERN = r'(?i)(%s)' % '|'.join([
    r'(?:\s|\()(' + re.escape(token) + r')\b'
    for token in DOI_TRUNCATE_AT_TOKENS
])


# https://jats.nlm.nih.gov/articleauthoring/tag-library/1.2/attribute/pub-id-type.html
KNOWN_PUB_ID_TYPES = {
    'accession',
    'archive',
    'ark',
    'art-access-id',
    'arxiv',
    'coden',
    'doaj',
    'doi',
    'handle',
    'index',
    'isbn',
    'manuscript',
    'medline',
    'mr',
    'other',
    'pii',
    'pmcid',
    'pmid',
    'publisher-id',
    'sici',
    'std-designation',
    'zbl'
}


def clone_node(node: etree.Element) -> etree.Element:
    return etree.fromstring(etree.tostring(node, encoding='unicode'))


def with_element_tail(element: etree.Element, tail: str) -> etree.Element:
    element.tail = tail
    return element


def get_jats_pub_id_element(text: str, pub_id_type: str, tail: str = None) -> etree.Element:
    node = E('pub-id', {'pub-id-type': pub_id_type}, text)
    if tail:
        node.tail = tail
    return node


def get_jats_doi_element(doi: str, **kwargs) -> etree.Element:
    return get_jats_pub_id_element(doi, 'doi', **kwargs)


def get_jats_pii_element(pii: str, **kwargs) -> etree.Element:
    return get_jats_pub_id_element(pii, 'pii', **kwargs)


def get_jats_pmid_element(pmid: str, **kwargs) -> etree.Element:
    return get_jats_pub_id_element(pmid, 'pmid', **kwargs)


def get_jats_pmcid_element(pmcid: str, **kwargs) -> etree.Element:
    return get_jats_pub_id_element(pmcid, 'pmcid', **kwargs)


def get_jats_other_pub_id_element(other_pub_id: str, **kwargs) -> etree.Element:
    return get_jats_pub_id_element(other_pub_id, 'other', **kwargs)


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


def add_next_sibling_element(element: etree.Element, new_element: etree.Element):
    element.getparent().insert(
        element.getparent().index(element) + 1,
        new_element
    )


def add_next_sibling_elements(element: etree.Element, new_elements: List[etree.Element]):
    for new_element in new_elements:
        add_next_sibling_element(element, new_element)
        element = new_element


def find_re_pattern_start_end(
        text: str,
        pattern: str,
        flags: int = 0,
        group_index: int = 1) -> Optional[Tuple[int, int]]:
    if text is None:
        raise RuntimeError('text requires, was none')
    m = re.search(pattern, text, flags=flags)
    if not m:
        LOGGER.debug('pattern (%r) not found in: %r', pattern, text)
        return None
    return m.start(group_index), m.end(group_index)


def remove_punct(text: str) -> str:
    return regex.sub(r'\p{Punct}', '', text)


def remove_punct_or_whitespace(text: str) -> str:
    return regex.sub(r'\p{Punct}|\s', '', text)


def strip_pii_from_doi(doi: str) -> str:
    if not doi.endswith('[pii]'):
        return doi
    doi = doi[0:-5].rstrip()
    doi_dup_token_candidate = doi.rsplit(' ', maxsplit=1)
    LOGGER.debug('doi_dup_token_candidate: %r', doi_dup_token_candidate)
    if len(doi_dup_token_candidate) != 2:
        return doi
    doi_start, dup_token_candidate = doi_dup_token_candidate
    if len(dup_token_candidate) < 3:
        # too short to be certain
        return doi
    if dup_token_candidate in doi_start:
        # obvious duplication of some part of the doi
        return doi_start.rstrip()
    doi_dup_token_candidate_no_punct = remove_punct(dup_token_candidate)
    if len(doi_dup_token_candidate_no_punct) < 3:
        # too short to be certain
        return doi
    doi_start_no_punct = remove_punct(doi_start)
    if doi_dup_token_candidate_no_punct in doi_start_no_punct:
        # repeat of part of the doi
        return doi_start.rstrip()
    return doi


def remove_duplicate_doi(doi: str) -> str:
    doi_prefix, path = doi.split('/', maxsplit=1)
    other_doi_start_end = find_re_pattern_start_end(path, DOI_PATTERN)
    if not other_doi_start_end:
        return doi
    other_doi_start, _ = other_doi_start_end
    other_doi = path[other_doi_start:]
    doi_start = doi_prefix + '/' + path[:other_doi_start]
    if other_doi in doi_start:
        return doi_start.rstrip()
    other_doi_no_punct = remove_punct_or_whitespace(other_doi)
    doi_start_no_punct = remove_punct_or_whitespace(doi_start)
    if other_doi_no_punct in doi_start_no_punct:
        return doi_start.rstrip()
    return doi


def truncate_doi_at_known_tokens(doi: str) -> str:
    m = re.search(DOI_TRUNCATE_AT_PATTERN, doi)
    LOGGER.debug(
        'truncate_doi_at_known_stop_words: doi=%r, p=%r, m=%s',
        doi, DOI_TRUNCATE_AT_PATTERN, m
    )
    if not m:
        return doi
    return doi[:m.start(1)].rstrip().rstrip('.')


def find_doi_start_end(text: str) -> Optional[Tuple[int, int]]:
    start_end = find_re_pattern_start_end(text, DOI_PATTERN)
    if start_end:
        start, end = start_end
        doi = text[start:end].rstrip().rstrip('.').rstrip()
        doi = truncate_doi_at_known_tokens(doi)
        if doi.endswith('[doi]'):
            doi = doi[0:-5].rstrip()
        doi = strip_pii_from_doi(doi)
        doi = remove_duplicate_doi(doi)
        doi = doi.rstrip(';')
        char_counts = Counter(doi)
        if char_counts[']'] > char_counts['[']:
            doi = doi.rstrip(']').rstrip()
        start_end = (start, start + len(doi))
        extracted_doi = text[start:start_end[1]]
        assert extracted_doi == doi, 'expect %r to be %r' % (extracted_doi, doi)
    return start_end


def find_doi_url_prefix_valid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, DOI_URL_PREFIX_PATTERN)


def find_pii_valid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PII_VALID_PATTERN)


def find_pii_other_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PII_OTHER_PATTERN)


def find_pmid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PMID_PATTERN)


def find_pmid_fix_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PMID_FIX_PATTERN)


def find_pmcid_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, PMCID_PATTERN, flags=re.IGNORECASE)


def find_wos_start_end(text: str) -> Optional[Tuple[int, int]]:
    return find_re_pattern_start_end(text, WOS_PATTERN, flags=re.IGNORECASE)


def find_doi_ext_link_start_end(text: str) -> Optional[Tuple[int, int]]:
    m = re.match(r'(.*)(\[' + DOI_PATTERN + r'\])', text)
    if not m:
        LOGGER.debug('not containing repeated doi: %r', text)
        return 0, len(text)
    return 0, m.start(2)


def find_ext_link_start_end(text: str) -> Optional[Tuple[int, int]]:
    if 'doi.org' in text:
        return find_doi_ext_link_start_end(text)
    LOGGER.debug('not doi ext link: %r', text)
    return 0, len(text)


def has_surrounding_quotes(text: str, start: int = 0, end: int = None) -> bool:
    if end is None:
        end = len(text)
    return (
        (end > start + 2)
        and (
            (text[start] == '"' and text[end - 1] == '"')
            or (text[start] == SpecialChars.LSQUO and text[end - 1] == SpecialChars.RSQUO)
            or (text[start] == SpecialChars.LDQUO and text[end - 1] == SpecialChars.RDQUO)
        )
    )


def find_article_title_start_end(text: str) -> Optional[Tuple[int, int]]:
    start_end = find_re_pattern_start_end(text, ARTICLE_TITLE_PATTERN)
    if not start_end:
        start_end = (0, len(text))
    start, end = start_end
    if has_surrounding_quotes(text, start, end):
        start += 1
        end -= 1
    return start, end


def change_annotation_to_matching_text(
        element: etree.Element,
        find_start_end_fn: Callable[[str], Optional[Tuple[int, int]]]):
    text = element.text
    if text is None:
        return
    start_end = find_start_end_fn(text)
    if not start_end:
        LOGGER.debug('%s not found in: %r', find_start_end_fn.__name__, text)
        replace_element_with_text(element, text)
        return
    start, end = start_end
    matching_text = text[start:end]
    LOGGER.debug('matching: %s (%r, %s)', start_end, matching_text, find_start_end_fn.__name__)
    element.text = matching_text
    add_text_to_previous(element, text[:start])
    add_text_to_tail_prefix(element, text[end:])


def change_annotations_to_matching_text(
        elements: List[etree.Element],
        *args,
        **kwargs):
    for element in elements:
        change_annotation_to_matching_text(element, *args, **kwargs)


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
        LOGGER.debug('%s not found in: %r', find_start_end_fn.__name__, text)
        return False
    start, end = start_end
    matching_text = text[start:end]
    LOGGER.debug('matching: %s (%r, %s)', start_end, matching_text, find_start_end_fn.__name__)
    element.text = text[:start]
    new_element = with_element_tail(
        create_element_fn(matching_text),
        tail=text[end:]
    )
    if as_next_sibling:
        add_next_sibling_element(element, new_element)
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
        LOGGER.debug('%s not found in: %r', find_start_end_fn.__name__, text)
        return False
    start, end = start_end
    matching_text = text[start:end]
    LOGGER.debug('matching: %s (%r, %s)', start_end, matching_text, find_start_end_fn.__name__)
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
    for mixed_citation_element in reference_element.xpath(JatsXpaths.MIXED_CITATION):
        if add_annotation_to_element_if_matching(
            mixed_citation_element,
            *args,
            **kwargs
        ):
            return True
    return False


def split_url(url: str) -> List[str]:
    pos = 0
    result = []
    for m in regex.finditer(r'https?://', url):
        start = m.start()
        if start > pos:
            result.append(url[pos:start])
        pos = start
    if len(url) > pos:
        result.append(url[pos:])
    return result


def fix_ext_link(reference_element: etree.Element):
    for child_element in list(reference_element.xpath(JatsXpaths.EXT_LINK)):
        text = child_element.text
        if not text:
            continue
        href = child_element.attrib.get(XLINK_HREF)
        hrefs = split_url(text)
        # very special case where hrefs are joined by 'w'
        if not href or (href != text and href != 'w'.join(hrefs)):
            continue
        LOGGER.debug('hrefs: %r', hrefs)
        if len(hrefs) > 1:
            child_element.text = hrefs[0]
            child_element.attrib[XLINK_HREF] = hrefs[0]
        add_next_sibling_elements(child_element, [
            get_jats_ext_link_element(other_href)
            for other_href in hrefs[1:]
        ])
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.EXT_LINK),
        find_start_end_fn=find_ext_link_start_end
    )
    for child_element in reference_element.xpath(JatsXpaths.EXT_LINK):
        href = child_element.attrib.get(XLINK_HREF)
        if not href:
            continue
        start_end = find_ext_link_start_end(href)
        if not start_end:
            continue
        start, end = start_end
        child_element.attrib[XLINK_HREF] = href[start:end]


def remove_surrounding_quotes_from_element(element: etree.Element):
    text = get_text_content(element)
    if len(text) < 2:
        return
    children = list(element)
    if has_surrounding_quotes(text):
        if element.text:
            add_text_to_previous(element, element.text[:1])
            element.text = element.text[1:]
        if children and children[-1].tail:
            add_text_to_tail_prefix(element, children[-1].tail[-1:])
            children[-1].tail = children[-1].tail[:-1]
    elif text[0] in LEFT_QUOTE_CHARS:
        right_quote_char = RIGHT_BY_LEFT_QUOTE_CHAR[text[0]]
        if right_quote_char not in text[1:] and element.text:
            add_text_to_previous(element, element.text[:1])
            element.text = element.text[1:]


def remove_training_comma_from_element(element: etree.Element):
    text = get_text_content(element)
    rstripped_text = text.rstrip(', ')
    if len(rstripped_text) == len(text):
        return
    children = list(element)
    if children and children[-1].tail:
        tail = children[-1].tail
        tail_end = max(0, len(tail) + len(rstripped_text) - len(text))
        add_text_to_tail_prefix(element, tail[tail_end:])
        children[-1].tail = tail[:tail_end]


def fix_article_title(reference_element: etree.Element):
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.ARTICLE_TITLE),
        find_start_end_fn=find_article_title_start_end
    )
    for element in reference_element.xpath(JatsXpaths.ARTICLE_TITLE):
        remove_surrounding_quotes_from_element(element)
        remove_training_comma_from_element(element)


def fix_doi(reference_element: etree.Element):
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.DOI),
        find_start_end_fn=find_doi_start_end
    )


def replace_doi_annotation_with_ext_link_if_url(reference_element: etree.Element):
    for doi_element in reference_element.xpath(JatsXpaths.DOI):
        previous_text = get_previous_text(doi_element)
        start_end = find_doi_url_prefix_valid_start_end(previous_text)
        if not start_end:
            LOGGER.debug('not matching doi url prefix: %r', previous_text)
            continue
        start, _ = start_end
        matching_doi_url_prefix = previous_text[start:]
        doi_url = matching_doi_url_prefix + doi_element.text
        LOGGER.debug('matching doi url prefix: %s (%r)', start_end, matching_doi_url_prefix)
        set_previous_text(doi_element, previous_text[:start])
        doi_element.getparent().replace(
            doi_element,
            get_jats_ext_link_element(
                doi_url,
                tail=doi_element.tail
            )
        )


def fix_pii(reference_element: etree.Element):
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.PII),
        find_start_end_fn=find_pii_valid_start_end
    )


def fix_pmid(reference_element: etree.Element):
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.PMID),
        find_start_end_fn=find_pmid_fix_start_end
    )


def fix_pmcid(reference_element: etree.Element):
    change_annotations_to_matching_text(
        reference_element.xpath(JatsXpaths.PMCID),
        find_start_end_fn=find_pmcid_start_end
    )


def add_doi_annotation_if_not_present(reference_element: etree.Element):
    if reference_element.xpath(JatsXpaths.DOI):
        return
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_doi_start_end,
        create_element_fn=get_jats_doi_element,
        parse_comment=False
    )


def add_pii_valid_annotation_if_not_present(reference_element: etree.Element):
    if reference_element.xpath(JatsXpaths.PII):
        return
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pii_valid_start_end,
        create_element_fn=get_jats_pii_element,
        parse_comment=False
    )


def add_pii_other_pub_id_annotation_if_not_present(reference_element: etree.Element):
    if reference_element.xpath(JatsXpaths.PII):
        return
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pii_other_start_end,
        create_element_fn=get_jats_other_pub_id_element,
        parse_comment=False
    )


def add_pmid_annotation_if_not_present(reference_element: etree.Element):
    if reference_element.xpath(JatsXpaths.PMID):
        return
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pmid_start_end,
        create_element_fn=get_jats_pmid_element,
        parse_comment=True
    )


def add_pmcid_annotation_if_not_present(reference_element: etree.Element):
    if reference_element.xpath(JatsXpaths.PMCID):
        return
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_pmcid_start_end,
        create_element_fn=get_jats_pmcid_element,
        parse_comment=True
    )


def add_wos_as_other_pub_id_annotation_if_not_present(reference_element: etree.Element):
    add_annotation_to_reference_element_if_matching(
        reference_element,
        find_start_end_fn=find_wos_start_end,
        create_element_fn=get_jats_other_pub_id_element,
        parse_comment=True
    )


def convert_known_pub_id_type_to_lower_case(reference_element: etree.Element):
    for pub_id_element in reference_element.xpath(JatsXpaths.PUB_ID):
        pub_id_type = pub_id_element.attrib.get('pub-id-type')
        if not pub_id_type:
            continue
        pub_id_type_lower = pub_id_type.lower()
        if pub_id_type_lower in KNOWN_PUB_ID_TYPES:
            pub_id_element.attrib['pub-id-type'] = pub_id_type_lower


def fix_reference(reference_element: etree.Element) -> etree.Element:
    convert_known_pub_id_type_to_lower_case(reference_element)
    fix_article_title(reference_element)
    fix_doi(reference_element)
    replace_doi_annotation_with_ext_link_if_url(reference_element)
    fix_ext_link(reference_element)
    fix_pii(reference_element)
    fix_pmid(reference_element)
    fix_pmcid(reference_element)
    add_pmid_annotation_if_not_present(reference_element)
    add_pmcid_annotation_if_not_present(reference_element)
    add_pii_valid_annotation_if_not_present(reference_element)
    add_pii_other_pub_id_annotation_if_not_present(reference_element)
    add_wos_as_other_pub_id_annotation_if_not_present(reference_element)
    add_doi_annotation_if_not_present(reference_element)
    return reference_element


def fix_jats_xml_node(root: etree.Element):
    for ref in root.xpath(JatsXpaths.REF):
        fix_reference(ref)
    return root


def iter_get_changed_original_references(
        root: etree.Element,
        original_root: etree.Element) -> Iterable[etree.Element]:
    original_ref_list = original_root.xpath(JatsXpaths.REF)
    ref_list = root.xpath(JatsXpaths.REF)
    assert len(ref_list) == len(original_ref_list)
    for ref, original_ref in zip(ref_list, original_ref_list):
        if etree.tostring(ref) == etree.tostring(original_ref):
            continue
        yield original_ref


def add_jats_meta_data(root: etree.Element, meta_data_dict: Dict[str, str]):
    if not meta_data_dict:
        return
    custom_meta_group = root.find('custom-meta-group')
    if custom_meta_group is None:
        custom_meta_group = with_element_tail(E('custom-meta-group', '\n'), '\n')
        root.insert(0, custom_meta_group)
    for key, value in meta_data_dict.items():
        custom_meta_group.append(with_element_tail(E(
            'custom-meta',
            E('meta-name', key),
            E('meta-value', value)
        ), tail='\n'))


def get_fix_xml_meta_data_dict(
        root: etree.Element,
        original_root: etree.Element,
        fixed_malformatted_xml: bool = False) -> Dict[str, str]:
    changed_original_references = list(iter_get_changed_original_references(
        root, original_root
    ))
    changed_original_reference_ids = [
        changed_original_reference.attrib.get('id', '')
        for changed_original_reference in changed_original_references
    ]
    return OrderedDict([
        (
            'fix-jats-note',
            'This XML file had been modified to fix some of the potential errors.'
        ), (
            'fix-jats-run-timestamp',
            datetime.utcnow().isoformat()
        ), (
            'fix-jats-malformatted-xml',
            str(fixed_malformatted_xml).lower()
        ), (
            'fix-jats-changed-reference-count',
            str(len(changed_original_references))
        ), (
            'fix-jats-changed-reference-ids',
            ','.join(changed_original_reference_ids)
        )
    ])


def add_fix_xml_meta_data(root: etree.Element, *args, **kwargs):
    add_jats_meta_data(root, get_fix_xml_meta_data_dict(
        root, *args, **kwargs
    ))


def fix_jats_xml_file(input_file: str, output_file: str, log_file_enabled: bool = True):
    if log_file_enabled:
        LOGGER.info('processing: %r -> %r', input_file, output_file)
    else:
        LOGGER.debug('processing: %r -> %r', input_file, output_file)
    fixed_malformatted_xml = False
    with auto_download_input_file(input_file) as local_input_file:
        try:
            tree = parse_xml(local_input_file, filename=input_file, fix_xml=False)
        except ValueError:
            tree = parse_xml(local_input_file, filename=input_file, fix_xml=True)
            fixed_malformatted_xml = True
    root = tree.getroot()
    original_root = clone_node(root)
    fix_jats_xml_node(root)
    add_fix_xml_meta_data(root, original_root, fixed_malformatted_xml=fixed_malformatted_xml)
    output_bytes = etree.tostring(
        tree,
        xml_declaration=True,
        encoding=tree.docinfo.encoding
    )
    save_file_content(output_file, output_bytes)


class FixJatsProcessor:
    def __init__(self, opt: argparse.Namespace):
        self.num_workers = opt.num_workers
        self.multi_processing = opt.multi_processing
        self.source_path = opt.source_path
        self.source_file_list_path = opt.source_file_list
        self.source_file_list_column = opt.source_file_list_column
        self.source_base_path = (
            opt.source_base_path
            or (self.source_file_list_path and os.path.dirname(self.source_file_list_path))
            or os.path.dirname(self.source_path)
        )
        LOGGER.debug('source_base_path: %s', self.source_base_path)
        self.output_path = opt.output_path
        self.source_filename_pattern = opt.source_filename_pattern
        self.limit = opt.limit
        self.log_file_enabled = not opt.no_log_file

    def get_output_file_for_source_file(self, source_url: str):
        return os.path.join(
            self.output_path,
            relative_path(self.source_base_path, source_url)
        )

    def process_source_file(self, source_file: str):
        try:
            output_file = self.get_output_file_for_source_file(source_file)
            assert output_file != source_file
            fix_jats_xml_file(source_file, output_file, log_file_enabled=self.log_file_enabled)
        except Exception as exc:
            raise RuntimeError('failed to process %r due to %r' % (source_file, exc)) from exc

    def process_source_file_serializable(self, source_file: str):
        try:
            return self.process_source_file(source_file)
        except Exception as exc:
            raise get_serializable_exception(exc) from exc

    def run_local_pipeline(self, xml_file_list: List[str]):
        num_workers = min(self.num_workers, len(xml_file_list))
        multi_processing = self.multi_processing
        LOGGER.info('using %d workers (multi_processing: %s)', num_workers, multi_processing)
        PoolExecutor = (
            concurrent.futures.ProcessPoolExecutor if multi_processing
            else concurrent.futures.ThreadPoolExecutor
        )
        process_source_file = (
            self.process_source_file_serializable if multi_processing
            else self.process_source_file
        )
        with PoolExecutor(max_workers=num_workers) as executor:
            with logging_tqdm(total=len(xml_file_list), logger=LOGGER) as pbar:
                future_to_url = {
                    executor.submit(process_source_file, url): url
                    for url in xml_file_list
                }
                LOGGER.debug('future_to_url: %s', future_to_url)
                for future in concurrent.futures.as_completed(future_to_url):
                    pbar.update(1)
                    future.result()

    def get_source_file_list(self):
        if self.source_file_list_path:
            return load_file_list(
                self.source_file_list_path,
                column=self.source_file_list_column,
                limit=self.limit
            )
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
    source_group.add_argument(
        '--source-file-list', type=str,
        help='path to source file list'
    )
    source_group.add_argument(
        '--source-file-list-column', type=str,
        default='xml_url',
        help='the column to use when reading the source file list (if csv or tsv)'
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

    parser.add_argument(
        '--no-log-file', action='store_true', default=False,
        help='disable logging of file being processed'
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
    if args.no_log_file:
        logging.getLogger('sciencebeam_utils.beam_utils.io').setLevel('WARNING')
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
