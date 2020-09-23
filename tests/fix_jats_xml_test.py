import logging
from pathlib import Path
from typing import List

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import (
    get_text_content,
    get_text_content_list
)

from sciencebeam_trainer_grobid_tools.utils.xml import (
    parse_xml
)

from sciencebeam_trainer_grobid_tools.fix_jats_xml import (
    XLINK_HREF,
    JatsXpaths,
    find_doi_start_end,
    find_doi_url_prefix_valid_start_end,
    find_pii_valid_start_end,
    get_jats_ext_link_element,
    get_jats_doi_element,
    get_jats_pii_element,
    get_jats_pmid_element,
    get_jats_pmcid_element,
    fix_reference as _fix_reference,
    main
)


LOGGER = logging.getLogger(__name__)


INVALID_PII_1 = '12/34/4567'
PII_1 = 'S0123-1234(11)01234-5'
DOI_1 = '10.12345/abc/1'
PMID_1 = '12345'
PMCID_1 = 'PMC1234567'

HTTPS_DOI_URL_PREFIX = 'https://doi.org/'
HTTPS_SPACED_DOI_URL_PREFIX = 'https : // doi . org / '


def get_jats_mixed_ref(*args) -> etree.Element:
    return E.ref(E('mixed-citation', *args))


def clone_node(node: etree.Element) -> etree.Element:
    return etree.fromstring(etree.tostring(node))


def get_jats(references: List[etree.Element]) -> etree.Element:
    return E.article(E.back(
        E('ref-list', *references)
    ))


def fix_reference(ref: etree.Element) -> etree.Element:
    original_ref_text = get_text_content(ref)
    LOGGER.debug('ref xml (before): %s', etree.tostring(ref))
    fixed_ref = _fix_reference(ref)
    LOGGER.debug('ref xml (after): %s', etree.tostring(fixed_ref))
    assert get_text_content(fixed_ref) == original_ref_text
    return fixed_ref


class TestFindDoiValidStartEnd:
    def test_should_find_valid_doi(self):
        text = 'before:  %s' % DOI_1
        start, end = find_doi_start_end(text)
        assert text[start:end] == DOI_1

    def test_should_allow_single_subdivision(self):
        doi = '10.1234.1/test'
        text = 'before:  %s' % doi
        start, end = find_doi_start_end(text)
        assert text[start:end] == doi

    def test_should_allow_multiple_subdivisions(self):
        doi = '10.1234.1.2.3/test'
        text = 'before:  %s' % doi
        start, end = find_doi_start_end(text)
        assert text[start:end] == doi

    def test_should_preserve_square_brackets(self):
        doi = DOI_1 + '[test]'
        text = 'before:  %s' % doi
        start, end = find_doi_start_end(text)
        assert text[start:end] == doi

    def test_should_ignore_square_brackets_around_doi(self):
        doi = DOI_1
        text = 'before:  [%s]' % doi
        start, end = find_doi_start_end(text)
        assert text[start:end] == doi

    def test_should_ignore_doi_square_brackets_label(self):
        doi = DOI_1
        text = 'before:  %s [doi]' % doi
        start, end = find_doi_start_end(text)
        assert text[start:end] == doi


class TestFindDoiUrlPrefixValidStartEnd:
    def test_should_find_https_doi_prefix(self):
        text = 'other:  https://doi.org/'
        start, end = find_doi_url_prefix_valid_start_end(text)
        assert text[start:end] == 'https://doi.org/'

    def test_should_find_http_doi_prefix(self):
        text = 'other:  http://doi.org/'
        start, end = find_doi_url_prefix_valid_start_end(text)
        assert text[start:end] == 'http://doi.org/'

    def test_should_find_https_dx_doi_prefix(self):
        text = 'other:  https://dx.doi.org/'
        start, end = find_doi_url_prefix_valid_start_end(text)
        assert text[start:end] == 'https://dx.doi.org/'


class TestFindPiiValidStartEnd:
    def test_should_accept_valid_pii(self):
        assert find_pii_valid_start_end(PII_1) is not None

    def test_should_not_accept_valid_pii(self):
        assert find_pii_valid_start_end(INVALID_PII_1) is None

    def test_should_accept_valid_pii_with_capital_x_with_punct(self):
        assert find_pii_valid_start_end('S0123-123X(11)01234-X') is not None

    def test_should_accept_valid_pii_with_capital_x_without_punct(self):
        assert find_pii_valid_start_end('S0123123X1101234X') is not None


class TestFixReference:
    def test_should_not_change_valid_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi_element(DOI_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_pub_id_element_if_not_containing_valid_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi_element('not a doi'))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == ''

    def test_should_convert_doi_with_inside_url_prefix_to_ext_link(self):
        original_ref = get_jats_mixed_ref(
            'some text',
            get_jats_doi_element(HTTPS_DOI_URL_PREFIX + DOI_1),
            'tail text'
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        ext_link_text = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.EXT_LINK)))
        assert ext_link_text == HTTPS_DOI_URL_PREFIX + DOI_1

    def test_should_convert_doi_with_outside_url_prefix_to_ext_link(self):
        original_ref = get_jats_mixed_ref(
            'some text ' + HTTPS_DOI_URL_PREFIX,
            get_jats_doi_element(DOI_1),
            'tail text'
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        ext_link_text = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.EXT_LINK)))
        assert ext_link_text == HTTPS_DOI_URL_PREFIX + DOI_1

    def test_should_convert_doi_with_outside_spaced_url_prefix_to_ext_link(self):
        original_ref = get_jats_mixed_ref(
            'some text ' + HTTPS_SPACED_DOI_URL_PREFIX,
            get_jats_doi_element(DOI_1),
            'tail text'
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        ext_links = fixed_ref.xpath(JatsXpaths.EXT_LINK)
        ext_link_text = '|'.join(get_text_content_list(ext_links))
        assert ext_link_text == HTTPS_SPACED_DOI_URL_PREFIX + DOI_1
        assert ext_links[0].attrib == {
            'ext-link-type': 'uri',
            XLINK_HREF: HTTPS_DOI_URL_PREFIX + DOI_1
        }

    def test_should_remove_doi_prefix_from_doi(self):
        original_ref = get_jats_mixed_ref('some text', get_jats_doi_element('doi:' + DOI_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_prefix_without_preceeding_text(self):
        original_ref = get_jats_mixed_ref(get_jats_doi_element('doi:' + DOI_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_prefix_after_preceeding_element_without_tail_text(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            get_jats_doi_element('doi:' + DOI_1)
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_prefix_after_preceeding_element_with_tail_text(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'tail text',
            get_jats_doi_element('doi:' + DOI_1)
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_suffix_from_doi_without_tail(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi_element(DOI_1 + ' [doi]')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_doi_suffix_from_doi_with_tail(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi_element(DOI_1 + ' [doi]'),
            'tail text'
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_remove_double_doi_in_ext_link_square_brackets(self):
        original_ref = get_jats_mixed_ref(
            get_jats_ext_link_element(HTTPS_DOI_URL_PREFIX + DOI_1 + '[' + DOI_1 + ']')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_ext_links = fixed_ref.xpath(JatsXpaths.EXT_LINK)
        fixed_ext_link = '|'.join(get_text_content_list(fixed_ext_links))
        assert fixed_ext_link == HTTPS_DOI_URL_PREFIX + DOI_1
        assert fixed_ext_links[0].attrib[XLINK_HREF] == HTTPS_DOI_URL_PREFIX + DOI_1

    def test_should_not_remove_other_square_brackets_from_ext_link(self):
        url = HTTPS_DOI_URL_PREFIX + DOI_1 + '[other]'
        original_ref = get_jats_mixed_ref(
            get_jats_ext_link_element(url)
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_ext_links = fixed_ref.xpath(JatsXpaths.EXT_LINK)
        fixed_ext_link = '|'.join(get_text_content_list(fixed_ext_links))
        assert fixed_ext_link == url
        assert fixed_ext_links[0].attrib[XLINK_HREF] == url

    def test_should_separately_annotate_pii_without_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi_element(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PII)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1

    def test_should_separately_annotate_pii_with_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'doi: ',
            get_jats_doi_element(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PII)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1

    def test_should_separately_annotate_invalid_pii_as_other_pub_id(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi_element(INVALID_PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        other_pub_id = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.OTHER_PUB_ID)))
        assert fixed_doi == DOI_1
        assert other_pub_id == INVALID_PII_1

    def test_should_remove_invalid_pii_pub_id(self):
        original_ref = get_jats_mixed_ref(
            get_jats_pii_element(INVALID_PII_1)
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PII)))
        assert fixed_pii == ''

    def test_should_not_include_doi_colon_in_pii(self):
        original_ref = get_jats_mixed_ref(
            'doi:',
            get_jats_doi_element(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PII)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1

    def test_should_annotate_missing_doi_with_label(self):
        original_ref = get_jats_mixed_ref('doi:' + DOI_1)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_annotate_missing_doi_excluding_dot(self):
        original_ref = get_jats_mixed_ref(DOI_1 + '.')
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_annotate_missing_doi_in_square_brackets(self):
        original_ref = get_jats_mixed_ref('[' + DOI_1 + ']')
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_keep_original_pmid_if_already_present_and_valid(self):
        original_ref = get_jats_mixed_ref(
            get_jats_pmid_element(PMID_1),
            ', alternative PMID: 123'
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_remove_pmid_non_digit_text(self):
        original_ref = get_jats_mixed_ref(
            get_jats_pmid_element('PMID: ' + PMID_1)
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_separately_annotate_pmid_without_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            'PMID:' + PMID_1
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_separately_annotate_pmid_with_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'PMID:' + PMID_1
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_separately_annotate_pmid_with_spaces(self):
        original_ref = get_jats_mixed_ref(
            ' PMID : ' + PMID_1 + ' '
        )
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_annotate_missing_pmid_in_comment(self):
        original_ref = get_jats_mixed_ref(E.comment('PMID:' + PMID_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMID)))
        assert fixed_pmid == PMID_1

    def test_should_remove_double_pmc_prefix_from_pmcid(self):
        original_ref = get_jats_mixed_ref('PMCID: ', get_jats_pmcid_element('PMC' + PMCID_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmcid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMCID)))
        assert fixed_pmcid == PMCID_1

    def test_should_annotate_missing_pmcid(self):
        original_ref = get_jats_mixed_ref('PMCID: ' + PMCID_1)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmcid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMCID)))
        assert fixed_pmcid == PMCID_1

    def test_should_annotate_missing_pmcid_in_comment(self):
        original_ref = get_jats_mixed_ref(E.comment(PMCID_1))
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_pmcid = '|'.join(get_text_content_list(fixed_ref.xpath(JatsXpaths.PMCID)))
        assert fixed_pmcid == PMCID_1


class TestMain:
    def test_should_fix_jats_xml_using_source_path(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi_element('doi:' + DOI_1))
        input_file = temp_dir / 'input' / 'file1.xml'
        input_file.parent.mkdir()
        input_file.write_bytes(etree.tostring(get_jats(references=[
            original_ref
        ])))
        output_file = temp_dir / 'output' / 'file1.xml'
        main([
            '--source-path=%s' % input_file,
            '--output-path=%s' % output_file.parent
        ])
        assert output_file.exists()
        fixed_root = parse_xml(str(output_file))
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_fix_jats_xml_using_source_base_path(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi_element('doi:' + DOI_1))
        input_file = temp_dir / 'input' / 'file1.xml'
        input_file.parent.mkdir()
        input_file.write_bytes(etree.tostring(get_jats(references=[
            original_ref
        ])))
        output_file = temp_dir / 'output' / 'file1.xml'
        main([
            '--source-base-path=%s' % input_file.parent,
            '--output-path=%s' % output_file.parent
        ])
        assert output_file.exists()
        fixed_root = parse_xml(str(output_file))
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1

    def test_should_fix_jats_xml_using_source_base_path_in_sub_directory(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi_element('doi:' + DOI_1))
        input_file = temp_dir / 'input' / 'sub' / 'file1.xml'
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(etree.tostring(get_jats(references=[
            original_ref
        ])))
        output_file = temp_dir / 'output' / 'sub' / 'file1.xml'
        main([
            '--source-base-path=%s' % input_file.parent,
            '--output-path=%s' % output_file.parent
        ])
        assert output_file.exists()
        fixed_root = parse_xml(str(output_file))
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(JatsXpaths.DOI)))
        assert fixed_doi == DOI_1
