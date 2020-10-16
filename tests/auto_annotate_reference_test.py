import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.xml import get_text_content

from sciencebeam_trainer_grobid_tools.utils.tei_xml import (
    TEI_E,
    get_tei_xpath_matches,
    get_tei_xpath_text
)

from sciencebeam_trainer_grobid_tools.auto_annotate_reference import main

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
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
ISSUE_2 = '8'
FIRST_PAGE_1 = '101'
LAST_PAGE_1 = '191'
ISSN_1 = '1012-4567'
ISBN_1 = '978-1-234-05432-5'
DOI_1 = '10.12345/test.2001.2.3'
PII_1 = 'S0123-1234(11)01234-5'
PMID_1 = '1234567'
PMCID_1 = 'PMC1000001'
ARXIV_1 = '1723.008484'
LINK_1 = 'https://test.org/path'


def get_reference_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return TEI_E.tei(TEI_E.text(TEI_E.back(TEI_E.listBibl(*items))))


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


def get_all_bibl(root: etree.Element, **kwargs) -> etree.Element:
    return get_tei_xpath_matches(root, '//tei:back/tei:listBibl/tei:bibl', **kwargs)


def get_first_bibl(root: etree.Element) -> etree.Element:
    return get_all_bibl(root, required=True)[0]


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_single_reference_with_single_field(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('article-title', ARTICLE_TITLE_1)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:title[@level="a"]') == (
            ARTICLE_TITLE_1
        )

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
            ', ;',
            E.volume(VOLUME_1),
            ' (',
            E.issue(ISSUE_1),
            '):',
            E.fpage(FIRST_PAGE_1),
            ', ISSN: ',
            E('issn', ISSN_1),
            ', ISBN: ',
            E('isbn', ISBN_1),
            ', doi: ',
            E('pub-id', DOI_1, {'pub-id-type': 'doi'}),
            ', pii: ',
            E('pub-id', PII_1, {'pub-id-type': 'pii'}),
            ', PMID: ',
            E('pub-id', PMID_1, {'pub-id-type': 'pmid'}),
            ', PMCID: ',
            E('pub-id', PMCID_1, {'pub-id-type': 'pmcid'}),
            ', arXiv: ',
            E('pub-id', ARXIV_1, {'pub-id-type': 'arXiv'}),
            ', web: ',
            E('ext-link', LINK_1),
            '.'
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:author') == (
            '%s, %s' % (LAST_NAME_1, FIRST_NAME_INITIAL_1)
        )
        assert get_tei_xpath_text(first_bibl, './tei:editor') == (
            '%s, %s' % (LAST_NAME_2, FIRST_NAME_INITIAL_2)
        )
        assert get_tei_xpath_text(first_bibl, './tei:date') == (
            YEAR_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:title[@level="a"]') == (
            ARTICLE_TITLE_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:title[@level="j"]') == (
            SOURCE_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:publisher') == (
            PUBLISHER_NAME_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:pubPlace') == (
            PUBLISHER_LOC_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="volume"]') == (
            VOLUME_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="issue"]') == (
            ISSUE_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="page"]') == (
            FIRST_PAGE_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="ISSN"]', '|') == (
            ISSN_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="ISBN"]', '|') == (
            ISBN_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="DOI"]', '|') == (
            DOI_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PII"]', '|') == (
            PII_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PMID"]', '|') == (
            PMID_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PMC"]', '|') == (
            PMCID_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="arxiv"]', '|') == (
            ARXIV_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:ptr[@type="web"]') == (
            LINK_1
        )

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
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:author', '|') == '%s, %s, %s, %s' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1,
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )

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
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:editor', '|') == '%s, %s, %s, %s' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1,
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )

    def test_should_include_etal_in_author_and_editor_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('string-name', E.surname(LAST_NAME_1), ', ', E('given-names', FIRST_NAME_INITIAL_1)),
            ' ',
            E.etal('et al.'),
            ' (',
            E.year(YEAR_1),
            ')',
            E('article-title', ARTICLE_TITLE_1),
            ' ',
            E(
                'person-group',
                {'person-group-type': 'editor'},
                E(
                    'string-name',
                    E.surname(LAST_NAME_2), ', ', E('given-names', FIRST_NAME_INITIAL_2)
                )
            ),
            ' ',
            E.etal('et al.')
        ]
        expected_author_string = '%s, %s et al.' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1
        )
        expected_editor_string = '%s, %s et al.' % (
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:author', '|') == expected_author_string
        assert get_tei_xpath_text(first_bibl, './tei:editor', '|') == expected_editor_string

    def test_should_include_dot_after_initials_in_author_and_editor_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('string-name', E.surname(LAST_NAME_1), ', ', E('given-names', FIRST_NAME_INITIAL_1)),
            '. ',
            ' (',
            E.year(YEAR_1),
            ')',
            E('article-title', ARTICLE_TITLE_1),
            ' ',
            E(
                'person-group',
                {'person-group-type': 'editor'},
                E(
                    'string-name',
                    E.surname(LAST_NAME_2), ', ', E('given-names', FIRST_NAME_INITIAL_2)
                )
            ),
            '. ',
            E.source(SOURCE_1),
        ]
        expected_author_string = '%s, %s.' % (
            LAST_NAME_1, FIRST_NAME_INITIAL_1
        )
        expected_editor_string = '%s, %s.' % (
            LAST_NAME_2, FIRST_NAME_INITIAL_2
        )
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:author', '|') == expected_author_string
        assert get_tei_xpath_text(first_bibl, './tei:editor', '|') == expected_editor_string

    def test_should_allow_varying_spaces_in_author_name(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('string-name', E.surname('Smith'), ', ', E('given-names', 'J. A')),
            ' (',
            E.year(YEAR_1),
            ') ',
            E('article-title', ARTICLE_TITLE_1)
        ]
        tei_author_text = 'Smith ,J .A .'
        tei_text = get_nodes_text([tei_author_text] + target_reference_content_nodes[1:])
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(tei_text)
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:author', '|') == tei_author_text

    def test_should_merge_multiple_issue_fields(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E.source(SOURCE_1),
            ', ',
            E.issue(ISSUE_1),
            '-',
            E.issue(ISSUE_2)
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="issue"]', '|') == (
            '%s-%s' % (ISSUE_1, ISSUE_2)
        )

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
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="page"]', '|') == (
            '%s-%s' % (FIRST_PAGE_1, LAST_PAGE_1)
        )

    def test_should_merge_multiple_page_fields_with_dot_suffix(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E.source(SOURCE_1),
            ', ',
            E.fpage(FIRST_PAGE_1),
            '-',
            E.lpage(LAST_PAGE_1),
            '.'
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="page"]', '|') == (
            '%s-%s' % (FIRST_PAGE_1, LAST_PAGE_1)
        )

    def test_should_annotate_with_same_issue_and_last_page_number(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        issue_number = '6'
        last_page = issue_number
        target_reference_content_nodes = [
            '(',
            E.issue(issue_number),
            '), ',
            E.fpage(FIRST_PAGE_1),
            '-',
            E.lpage(last_page),
            '.'
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="issue"]', '|') == (
            issue_number
        )
        assert get_tei_xpath_text(first_bibl, './tei:biblScope[@unit="page"]', '|') == (
            '%s-%s' % (FIRST_PAGE_1, last_page)
        )

    def test_should_add_idno_prefix_if_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('article-title', ARTICLE_TITLE_1),
            ', ISSN: ',
            E('issn', ISSN_1),
            ', ISBN: ',
            E('isbn', ISBN_1),
            ', doi: ',
            E('pub-id', DOI_1, {'pub-id-type': 'doi'}),
            ', pii: ',
            E('pub-id', PII_1, {'pub-id-type': 'pii'}),
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
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(
            first_bibl, './tei:idno[@type="ISSN"]', '|'
        ) == (
            'ISSN: ' + ISSN_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="ISBN"]', '|') == (
            'ISBN: ' + ISBN_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="DOI"]', '|') == (
            'doi: ' + DOI_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PII"]', '|') == (
            'pii: ' + PII_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PMID"]', '|') == (
            'PMID: ' + PMID_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PMC"]', '|') == (
            'PMCID: ' + PMCID_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="arxiv"]', '|') == (
            'arXiv: ' + ARXIV_1
        )

    def test_should_not_add_unrelated_idno_prefix_if_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('article-title', ARTICLE_TITLE_1),
            ', doi: ',
            E('pub-id', PII_1, {'pub-id-type': 'pii'}),
            '[pii]',
            E('pub-id', DOI_1, {'pub-id-type': 'doi'}),
            '[doi]',
            ', PMID: ',
            E('pub-id', PMID_1, {'pub-id-type': 'pmid'})
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(get_nodes_text(target_reference_content_nodes))
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="DOI"]', '|') == (
            DOI_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PII"]', '|') == (
            PII_1
        )
        assert get_tei_xpath_text(first_bibl, './tei:idno[@type="PMID"]', '|') == (
            'PMID: ' + PMID_1
        )

    def test_should_not_preserve_original_bibl_segmentation_when_segmenting_references(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats reference that we expect to be reflected
        # in the updated tei refererence tagging
        prefix = ARTICLE_TITLE_1
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        invalid_text = 'invalid reference'
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, jats_text, jats_text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(tei_text),
                TEI_E.bibl(tei_text),
                TEI_E.bibl(invalid_text)
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference',
            'segment-references': True
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_tei_xpath_text(tei_auto_root, './/tei:bibl', '|') == (
            tei_text + tei_text
        )
        assert get_tei_xpath_text(tei_auto_root, './/tei:note', '|') == (
            invalid_text
        )

    def test_should_remove_invalid_references_if_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats reference that we expect to be reflected
        # in the updated tei refererence tagging
        prefix = ARTICLE_TITLE_1
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        invalid_text = 'invalid reference'
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, jats_text, jats_text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(tei_text),
                TEI_E.bibl(tei_text),
                TEI_E.bibl(invalid_text)
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'reference',
            'segment-references': True,
            'remove-invalid-references': True
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_tei_xpath_text(tei_auto_root, './/tei:bibl', '|') == (
            tei_text + tei_text
        )
        assert get_tei_xpath_text(tei_auto_root, './/tei:note', '|') == ''

    def test_should_preserve_original_bibl_segmentation_when_not_segmenting_references(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats reference that would usually change the tei refererences
        # but for the references we expect to retain the original bibl segmentation
        prefix = ARTICLE_TITLE_1
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, jats_text, jats_text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(tei_text),
                TEI_E.bibl(tei_text)
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
        assert get_tei_xpath_text(tei_auto_root, './/tei:bibl', '|') == '|'.join([
            tei_text,
            tei_text
        ])

    def test_should_preserve_original_bibl_element_with_single_immediately_child(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        text = ARTICLE_TITLE_1
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(TEI_E.ptr(text)),
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
        bibl_list = get_all_bibl(tei_auto_root)
        assert (
            [get_text_content(bibl) for bibl in bibl_list]
            == [text]
        )

    def test_should_not_preserve_sub_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        text_1 = ARTICLE_TITLE_1
        text_2 = FIRST_PAGE_1
        text = text_1 + ' ' + text_2
        target_reference_content_nodes = [
            text
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, *target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_reference_tei_node([
                TEI_E.bibl(text_1, ' ', TEI_E.idno(text_2)),
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
        first_bibl = get_first_bibl(tei_auto_root)
        assert get_text_content(first_bibl) == text
        assert get_tei_xpath_text(first_bibl, './tei:idno', '|') == ''
