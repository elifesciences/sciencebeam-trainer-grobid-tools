import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import ElementMaker, E

from sciencebeam_utils.utils.xml import get_text_content, get_text_content_list

from sciencebeam_trainer_grobid_tools.utils.tei_xml import TEI_NS, TEI_NS_MAP, tei_xpath

from sciencebeam_trainer_grobid_tools.auto_annotate_affiliation_address import main

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_xpath_text,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.header.affiliation.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).header.affiliation.tei.xml/\1.xml/'

AFFILIATION_XPATH = (
    './tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic'
    '/tei:author/tei:affiliation'
)

TEXT_1 = 'text 1'

LABEL_1 = '1'

TEI_E = ElementMaker(namespace=TEI_NS, nsmap=TEI_NS_MAP)


def _get_first_tei_xpath(parent: etree.Element, xpath: str) -> List[etree.Element]:
    result = tei_xpath(parent, xpath)
    if not result:
        xpath_fragments = xpath.split('/')
        for fragment_count in reversed(range(1, len(xpath_fragments))):
            parent_xpath = '/'.join(xpath_fragments[:fragment_count])
            if len(parent_xpath) <= 1:
                break
            parent_result = tei_xpath(parent, parent_xpath)
            if parent_result:
                LOGGER.debug(
                    'no results for %r, but found matching elements for %r: %s',
                    xpath, parent_xpath, parent_result
                )
                break
        raise ValueError('no item found for xpath: %r (in %s)' % (xpath, parent))
    return result[0]


def get_tei_xpath_text(*args, **kwargs):
    return get_xpath_text(*args, namespaces=TEI_NS_MAP, **kwargs)


def get_affiliation_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return TEI_E.tei(TEI_E.teiHeader(TEI_E.fileDesc(TEI_E.sourceDesc(TEI_E.biblStruct(
        TEI_E.analytic(TEI_E.author(*items))
    )))))


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


def get_nodes_text(nodes: List[Union[str, etree.Element]]) -> str:
    return ''.join([
        str(node)
        if isinstance(node, str)
        else get_text_content(node)
        for node in nodes
    ])


def get_all_affiliations(root: etree.Element) -> etree.Element:
    return tei_xpath(root, AFFILIATION_XPATH)


def get_first_affiliation(root: etree.Element) -> etree.Element:
    return _get_first_tei_xpath(root, AFFILIATION_XPATH)


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_single_affiliation_with_single_field(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_reference_content_nodes = [
            E('label', LABEL_1),
            ' Some text'
        ]
        target_jats_xml = etree.tostring(
            get_target_xml_node(affiliation_nodes=[
                E.aff(*target_reference_content_nodes),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_affiliation_tei_node([
                TEI_E.affiliation(get_nodes_text(target_reference_content_nodes))
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'author_aff'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        first_aff = get_first_affiliation(tei_auto_root)
        assert get_tei_xpath_text(first_aff, './tei:marker') == (
            LABEL_1
        )

    @log_on_exception
    def test_should_preserve_original_affiliation_annotation(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats affiliation that would usually change the tei affiliation
        # without --segment-affiliation, we expect to retain the original affiliation segmentation
        prefix = 'Some affiliation'
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        target_jats_xml = etree.tostring(
            get_target_xml_node(affiliation_nodes=[
                E.aff(jats_text, jats_text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_affiliation_tei_node([
                TEI_E.affiliation(tei_text),
                TEI_E.affiliation(tei_text)
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'author_aff'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_text_content_list(get_all_affiliations(tei_auto_root)) == [
            tei_text,
            tei_text
        ]

    @log_on_exception
    def test_should_not_preserve_original_affiliation_annotation(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats affiliation that would usually change the tei affiliation
        # with --segment-affiliation, we expect to the affiliation segmentation to be updated
        prefix = 'Some affiliation'
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        target_jats_xml = etree.tostring(
            get_target_xml_node(affiliation_nodes=[
                E.aff(jats_text, jats_text),
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_affiliation_tei_node([
                TEI_E.affiliation(tei_text),
                TEI_E.affiliation(tei_text)
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'author_aff',
            'segment-affiliation': True
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_text_content_list(get_all_affiliations(tei_auto_root)) == [
            tei_text + tei_text
        ]

    @log_on_exception
    def test_should_remove_invalid_affiliation(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        # we only create a single jats affiliation that would usually change the tei affiliation
        # with --segment-affiliation, we expect to the affiliation segmentation to be updated
        prefix = 'Some affiliation'
        jats_text = prefix + '.'
        tei_text = prefix + ' .'
        invalid_affiliation_text = 'invalid affiliation'
        target_jats_xml = etree.tostring(
            get_target_xml_node(affiliation_nodes=[
                E.aff(jats_text)
            ])
        )
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_affiliation_tei_node([
                TEI_E.affiliation(tei_text),
                TEI_E.affiliation(invalid_affiliation_text)
            ])
        ))
        LOGGER.debug('target_jats_xml: %s', target_jats_xml)
        test_helper.xml_file_path.write_bytes(target_jats_xml)
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': 'simple',
            'fields': 'author_aff',
            'segment-affiliation': True,
            'remove-invalid-affiliations': True
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_text_content(tei_auto_root) == tei_text
