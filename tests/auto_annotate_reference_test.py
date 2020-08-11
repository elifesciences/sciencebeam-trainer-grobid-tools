import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content

from sciencebeam_trainer_grobid_tools.auto_annotate_reference import (
    main
)

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_xpath_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.references.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).references.tei.xml/\1.xml/'

TEXT_1 = 'text 1'

LABEL_1 = '1'

ARTICLE_TITLE_1 = 'article title A'
SOURCE_1 = 'source A'
PUBLISHER_NAME_1 = 'Publisher A'
PUBLISHER_LOC_1 = 'New London'
LAST_NAME_1 = 'Smith'
FIRST_NAME_INITIAL_1 = 'A'
LAST_NAME_2 = 'Johnson'
FIRST_NAME_INITIAL_2 = 'B'
YEAR_1 = '2001'
VOLUME_1 = '11'
ISSUE_1 = '7'
FIRST_PAGE_1 = '101'
LAST_PAGE_1 = '191'
ISSN_1 = '1012-4567'
DOI_1 = '10.12345/test.2001.2.3'
PMID_1 = '1234567'
PMCID_1 = 'PMC1000001'
ARXIV_1 = '1723.008484'
LINK_1 = 'https://test.org/path'


def get_reference_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(E.back(E.listBibl(*items))))


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


def get_jats_reference_node(
        label: str,
        *children: List[Union[str, etree.Element]]) -> etree.Element:
    ref = E.ref()
    if label:
        ref.append(E.label(label))
    ref.append(E('mixed-citation', *children))
    return ref


def get_nodes_text(nodes: List[Union[str, etree.Element]]) -> str:
    return ''.join([
        str(node)
        if isinstance(node, str)
        else get_text_content(node)
        for node in nodes
    ])


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_single_reference_with_all_fields(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('string-name', E.surname(LAST_NAME_1), ', ', E('given-names', FIRST_NAME_INITIAL_1)),
            ' (',
            E.year(YEAR_1),
            ')',
            E('article-title', ARTICLE_TITLE_1),
            ' In: ',
            E(
                'person-group',
                {'person-group-type': 'editor'},
                E(
                    'string-name',
                    E.surname(LAST_NAME_2), ', ', E('given-names', FIRST_NAME_INITIAL_2)
                )
            ),
            ' ',
            E.source(SOURCE_1),
            ' ',
            E('publisher-name', PUBLISHER_NAME_1),
            ', ',
            E('publisher-loc', PUBLISHER_LOC_1),
            ', ',
            E.volume(VOLUME_1),
            ' (',
            E.issue(ISSUE_1),
            '):',
            E.fpage(FIRST_PAGE_1),
            ', ISSN: ',
            E('issn', ISSN_1),
            ', doi: ',
            E('pub-id', DOI_1, {'pub-id-type': 'doi'}),
            ', PMID: ',
            E('pub-id', PMID_1, {'pub-id-type': 'pmid'}),
            ', PMCID: ',
            E('pub-id', PMCID_1, {'pub-id-type': 'pmcid'}),
            ', arXiv: ',
            E('pub-id', ARXIV_1, {'pub-id-type': 'arXiv'}),
            ', web: ',
            E('ext-link', LINK_1)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                E.bibl(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_bibl = tei_auto_root.xpath('//listBibl/bibl[1]')[0]
        assert get_xpath_text(first_bibl, './author') == (
            '%s, %s' % (LAST_NAME_1, FIRST_NAME_INITIAL_1)
        )
        assert get_xpath_text(first_bibl, './editor') == (
            '%s, %s' % (LAST_NAME_2, FIRST_NAME_INITIAL_2)
        )
        assert get_xpath_text(first_bibl, './date') == (
            YEAR_1
        )
        assert get_xpath_text(first_bibl, './title[@level="a"]') == (
            ARTICLE_TITLE_1
        )
        assert get_xpath_text(first_bibl, './title[@level="j"]') == (
            SOURCE_1
        )
        assert get_xpath_text(first_bibl, './publisher') == (
            PUBLISHER_NAME_1
        )
        assert get_xpath_text(first_bibl, './pubPlace') == (
            PUBLISHER_LOC_1
        )
        assert get_xpath_text(first_bibl, './biblScope[@unit="volume"]') == (
            VOLUME_1
        )
        assert get_xpath_text(first_bibl, './biblScope[@unit="issue"]') == (
            ISSUE_1
        )
        assert get_xpath_text(first_bibl, './biblScope[@unit="page"]') == (
            FIRST_PAGE_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="ISSN"]', '|') == (
            ISSN_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="DOI"]', '|') == (
            DOI_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="PMID"]', '|') == (
            PMID_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="PMC"]', '|') == (
            PMCID_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="arxiv"]', '|') == (
            ARXIV_1
        )
        assert get_xpath_text(first_bibl, './ptr[@type="web"]') == (
            LINK_1
        )

    @log_on_exception
    def test_should_merge_multiple_author_fields(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('string-name', E.surname(LAST_NAME_1), ', ', E('given-names', FIRST_NAME_INITIAL_1)),
            ', ',
            E('string-name', E.surname(LAST_NAME_2), ', ', E('given-names', FIRST_NAME_INITIAL_2)),
            ' (',
            E.year(YEAR_1),
            ')',
            E('article-title', ARTICLE_TITLE_1)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                E.bibl(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_bibl = tei_auto_root.xpath('//listBibl/bibl[1]')[0]
        assert get_xpath_text(first_bibl, './author', '|') == '%s, %s, %s, %s' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1,
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )

    @log_on_exception
    def test_should_merge_multiple_editor_fields(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E(
                'person-group',
                {'person-group-type': 'editor'},
                E(
                    'string-name',
                    E.surname(LAST_NAME_1), ', ', E('given-names', FIRST_NAME_INITIAL_1)
                ),
                ', ',
                E(
                    'string-name',
                    E.surname(LAST_NAME_2), ', ', E('given-names', FIRST_NAME_INITIAL_2)
                )
            ),
            ' (',
            E.year(YEAR_1),
            ')',
            E('article-title', ARTICLE_TITLE_1)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                E.bibl(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_bibl = tei_auto_root.xpath('//listBibl/bibl[1]')[0]
        assert get_xpath_text(first_bibl, './editor', '|') == '%s, %s, %s, %s' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1,
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )

    @log_on_exception
    def test_should_merge_multiple_page_fields(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E.source(SOURCE_1),
            ', ',
            E.fpage(FIRST_PAGE_1),
            '-',
            E.lpage(LAST_PAGE_1)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                E.bibl(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_bibl = tei_auto_root.xpath('//listBibl/bibl[1]')[0]
        assert get_xpath_text(first_bibl, './biblScope[@unit="page"]', '|') == (
            '%s-%s' % (FIRST_PAGE_1, LAST_PAGE_1)
        )

    @log_on_exception
    def test_should_add_idno_prefix_if_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('article-title', ARTICLE_TITLE_1),
            ', ISSN: ',
            E('issn', ISSN_1),
            ', doi: ',
            E('pub-id', DOI_1, {'pub-id-type': 'doi'}),
            ', PMID: ',
            E('pub-id', PMID_1, {'pub-id-type': 'pmid'}),
            ', PMCID: ',
            E('pub-id', PMCID_1, {'pub-id-type': 'pmcid'}),
            ', arXiv: ',
            E('pub-id', ARXIV_1, {'pub-id-type': 'arXiv'}),
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                E.bibl(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference',
            'include-idno-prefix': True
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_bibl = tei_auto_root.xpath('//listBibl/bibl[1]')[0]
        assert get_xpath_text(first_bibl, './idno[@type="ISSN"]', '|') == (
            'ISSN: ' + ISSN_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="DOI"]', '|') == (
            'doi: ' + DOI_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="PMID"]', '|') == (
            'PMID: ' + PMID_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="PMC"]', '|') == (
            'PMCID: ' + PMCID_1
        )
        assert get_xpath_text(first_bibl, './idno[@type="arxiv"]', '|') == (
            'arXiv: ' + ARXIV_1
        )
