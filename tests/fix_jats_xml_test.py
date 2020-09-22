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
    fix_reference,
    main
)


LOGGER = logging.getLogger(__name__)


PII_1 = '12/34/4567'
DOI_1 = '10.12345/abc/1'
PMID_1 = '12345'

HTTPS_DOI_URL_PREFIX = 'https://doi.org/'

DOI_XPATH = './/pub-id[@pub-id-type="doi"]'
PII_XPATH = './/pub-id[@pub-id-type="pii"]'
PMID_XPATH = './/pub-id[@pub-id-type="pmid"]'


def get_jats_mixed_ref(*args) -> etree.Element:
    return E.ref(E('mixed-citation', *args))


def get_jats_doi(doi: str) -> etree.Element:
    return E('pub-id', {'pub-id-type': 'doi'}, doi)


def get_jats_pmid(pmid: str) -> etree.Element:
    return E('pub-id', {'pub-id-type': 'pmid'}, pmid)


def clone_node(node: etree.Element) -> etree.Element:
    return etree.fromstring(etree.tostring(node))


def get_jats(references: List[etree.Element]) -> etree.Element:
    return E.article(E.back(
        E('ref-list', *references)
    ))


class TestFixReference:
    def test_should_not_change_valid_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_pub_id_element_if_not_containing_valid_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi('not a doi'))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == ''
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_from_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_without_preceeding_text(self):
        original_ref = get_jats_mixed_ref(get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_after_preceeding_element_without_tail_text(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1)
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_after_preceeding_element_with_tail_text(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'tail text',
            get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1)
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_suffix_from_doi_without_tail(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi(DOI_1 + ' [doi]')
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_suffix_from_doi_with_tail(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi(DOI_1 + ' [doi]'),
            'tail text'
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_separately_annotate_pii_without_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(PII_XPATH)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_separately_annotate_pii_with_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'doi: ',
            get_jats_doi(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(PII_XPATH)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_not_include_doi_colon_in_pii(self):
        original_ref = get_jats_mixed_ref(
            'doi:',
            get_jats_doi(PII_1 + ' [pii]; ' + DOI_1 + ' [doi]')
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        fixed_pii = '|'.join(get_text_content_list(fixed_ref.xpath(PII_XPATH)))
        assert fixed_doi == DOI_1
        assert fixed_pii == PII_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_keep_original_pmid_if_already_present_and_valid(self):
        original_ref = get_jats_mixed_ref(
            get_jats_pmid(PMID_1),
            ', alternative PMID: 123'
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(PMID_XPATH)))
        assert fixed_pmid == PMID_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_separately_annotate_pmid_without_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            'PMID:' + PMID_1
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(PMID_XPATH)))
        assert fixed_pmid == PMID_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_separately_annotate_pmid_with_preceding_element(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            'PMID:' + PMID_1
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(PMID_XPATH)))
        assert fixed_pmid == PMID_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_separately_annotate_pmid_with_spaces(self):
        original_ref = get_jats_mixed_ref(
            ' PMID : ' + PMID_1 + ' '
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_reference(clone_node(original_ref))
        LOGGER.debug('ref: %s', etree.tostring(fixed_ref))
        fixed_pmid = '|'.join(get_text_content_list(fixed_ref.xpath(PMID_XPATH)))
        assert fixed_pmid == PMID_1
        assert get_text_content(fixed_ref) == original_ref_text


class TestMain:
    def test_should_fix_jats_xml_using_source_path(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
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
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1

    def test_should_fix_jats_xml_using_source_base_path(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
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
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1

    def test_should_fix_jats_xml_using_source_base_path_in_sub_directory(self, temp_dir: Path):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
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
        fixed_doi = '|'.join(get_text_content_list(fixed_root.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
