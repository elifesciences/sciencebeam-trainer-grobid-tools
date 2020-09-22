from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import (
    get_text_content,
    get_text_content_list
)

from sciencebeam_trainer_grobid_tools.fix_jats_xml import (
    fix_doi
)


DOI_1 = '10.12345/abc/1'

HTTPS_DOI_URL_PREFIX = 'https://doi.org/'

DOI_XPATH = './/pub-id[@pub-id-type="doi"]'


def get_jats_mixed_ref(*args) -> etree.Element:
    return E.ref(E('mixed-citation', *args))


def get_jats_doi(doi: str) -> etree.Element:
    return E('pub-id', {'pub-id-type': 'doi'}, doi)


def clone_node(node: etree.Element) -> etree.Element:
    return etree.fromstring(etree.tostring(node))


class TestFixDoi:
    def test_should_not_change_valid_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_doi(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_from_doi(self):
        original_ref = get_jats_mixed_ref('doi: ', get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_doi(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_without_preceeding_text(self):
        original_ref = get_jats_mixed_ref(get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1))
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_doi(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_url_prefix_after_preceeding_element_without_tail_text(self):
        original_ref = get_jats_mixed_ref(
            E.other('other text'),
            get_jats_doi(HTTPS_DOI_URL_PREFIX + DOI_1)
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_doi(clone_node(original_ref))
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
        fixed_ref = fix_doi(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text

    def test_should_remove_doi_suffix_from_doi_without_tail(self):
        original_ref = get_jats_mixed_ref(
            'doi: ',
            get_jats_doi(DOI_1 + ' [doi]')
        )
        original_ref_text = get_text_content(original_ref)
        fixed_ref = fix_doi(clone_node(original_ref))
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
        fixed_ref = fix_doi(clone_node(original_ref))
        fixed_doi = '|'.join(get_text_content_list(fixed_ref.xpath(DOI_XPATH)))
        assert fixed_doi == DOI_1
        assert get_text_content(fixed_ref) == original_ref_text
