import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.utils.xml import get_xpath_text

from sciencebeam_trainer_grobid_tools.auto_annotate_header import (
    main
)

from .test_utils import log_on_exception, dict_to_args
from .auto_annotate_test_utils import (
    get_target_xml_node,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.header.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).header.tei.xml/\1.xml/'

TEXT_1 = 'text 1'

TITLE_1 = 'Chocolate bars for mice'
ABSTRACT_PREFIX_1 = 'Abstract'
ABSTRACT_1 = (
    'This study explores the nutritious value of chocolate bars for mice.'
)
NOT_MATCHING_ABSTRACT_1 = (
    'Something different.'
)


def get_header_tei_node(
        front_items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(E.front(*front_items)))


def get_default_tei_node() -> etree.Element:
    return get_header_tei_node([E.note(TEXT_1)])


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_title(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(TEXT_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TEXT_1)
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TEXT_1

    def test_should_auto_annotate_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(TEXT_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TEXT_1)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TEXT_1

    def test_should_extend_title_annotation_to_whole_line(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        title_text = 'Chocolate bars for mice'
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note('Title: ' + title_text)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=title_text)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == title_text

    def test_should_auto_annotate_multiple_fields_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        author_text = 'Mary Maison 1, John Smith 1'
        affiliation_text = '1 University of Science, Smithonia'
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([
                E.note(TITLE_1), E.lb(),
                E.note(author_text), E.lb(),
                E.note(affiliation_text), E.lb(),
                E.note(ABSTRACT_PREFIX_1, E.lb(), ABSTRACT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            title=TITLE_1,
            author_nodes=[
                E.contrib(E.name(
                    E.surname('Maison'),
                    E('given-names', 'Mary')
                )),
                E.contrib(E.name(
                    E.surname('Smith'),
                    E('given-names', 'John')
                )),
                E.aff(
                    E.institution('University of Science'),
                    E.country('Smithonia')
                )
            ],
            abstract_node=E.abstract(E.p(ABSTRACT_1))
        )))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join(['title', 'author', 'author_aff', 'abstract']),
            'matcher': 'simple'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TITLE_1
        assert get_xpath_text(tei_auto_root, '//byline/docAuthor') == author_text
        assert get_xpath_text(tei_auto_root, '//byline/affiliation') == affiliation_text
        assert get_xpath_text(tei_auto_root, '//div[@type="abstract"]') == (
            ABSTRACT_PREFIX_1 + ABSTRACT_1
        )

    def test_should_replace_affiliation_with_author_if_single_tokens(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        author_text = 'Mary Maison 1, John Smith 1'
        affiliation_text = '1 University of Science, Smithonia'
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([
                E.note(author_text), E.lb(),
                E.note(affiliation_text), E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            author_nodes=[
                E.contrib(E.name(
                    E.surname('Maison'),
                    E('given-names', 'Mary')
                )),
                E.contrib(E.name(
                    E.surname('Smith'),
                    E('given-names', 'John')
                )),
                E.aff(
                    E.label('1'),
                    E.institution('University of Science'),
                    E.country('Smithonia')
                )
            ]
        )))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join(['title', 'author', 'author_aff', 'abstract']),
            'matcher': 'simple'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//byline/docAuthor') == author_text
        assert get_xpath_text(tei_auto_root, '//byline/affiliation') == affiliation_text

    @pytest.mark.skip(
        reason='difficult to implement correctly due to prefix only seeging untagged text'
    )
    def test_should_auto_annotate_affiliation_preceding_number_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        author_text = 'Mary Maison 1, John Smith 1'
        affiliation_text_1 = '1'
        affiliation_text_2 = 'University of Science, Smithonia'
        affiliation_text = ' '.join([affiliation_text_1, affiliation_text_2])
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([
                E.note(TITLE_1), E.lb(),
                E.note(author_text), E.lb(),
                E.note(affiliation_text_1), E.lb(),
                E.note(affiliation_text_2), E.lb(),
                E.note(ABSTRACT_PREFIX_1, E.lb(), ABSTRACT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            title=TITLE_1,
            author_nodes=[
                E.contrib(E.name(
                    E.surname('Maison'),
                    E('given-names', 'Mary')
                )),
                E.contrib(E.name(
                    E.surname('Smith'),
                    E('given-names', 'John')
                )),
                E.aff(
                    E.institution('University of Science'),
                    E.country('Smithonia')
                )
            ],
            abstract_node=E.abstract(E.p(ABSTRACT_1))
        )))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join(['title', 'author', 'author_aff', 'abstract']),
            'matcher': 'simple'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TITLE_1
        assert get_xpath_text(tei_auto_root, '//byline/docAuthor') == author_text
        assert get_xpath_text(tei_auto_root, '//byline/affiliation') == affiliation_text
        assert get_xpath_text(tei_auto_root, '//div[@type="abstract"]') == (
            ABSTRACT_PREFIX_1 + ABSTRACT_1
        )

    def test_should_auto_annotate_alternative_spellings_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        author_text = 'Mary Maison 1, John Smith 1'
        affiliation_text = 'Berkeley, CA 12345, USA'
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([
                E.note(TITLE_1), E.lb(),
                E.note(author_text), E.lb(),
                E.note(affiliation_text), E.lb(),
                E.note(ABSTRACT_PREFIX_1, E.lb(), ABSTRACT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            title=TITLE_1,
            author_nodes=[
                E.contrib(E.name(
                    E.surname('Maison'),
                    E('given-names', 'Mary')
                )),
                E.contrib(E.name(
                    E.surname('Smith'),
                    E('given-names', 'John')
                )),
                E.aff(
                    E.institution('Berkeley'),
                    E.country('United States')
                )
            ],
            abstract_node=E.abstract(E.p(ABSTRACT_1))
        )))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join(['title', 'author', 'author_aff', 'abstract']),
            'matcher': 'simple'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TITLE_1
        assert get_xpath_text(tei_auto_root, '//byline/docAuthor') == author_text
        assert get_xpath_text(tei_auto_root, '//byline/affiliation') == affiliation_text
        assert get_xpath_text(tei_auto_root, '//div[@type="abstract"]', '|') == (
            ABSTRACT_PREFIX_1 + ABSTRACT_1
        )

    def test_should_skip_errors(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        tei_raw_other_file_path = test_helper.tei_raw_path.joinpath('document0.header.tei.xml')
        tei_raw_other_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(TEXT_1)])
        ))
        xml_other_file_path = test_helper.xml_path.joinpath('document0.xml')
        xml_other_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TEXT_1)
        ) + b'error')
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([E.note(TEXT_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TEXT_1)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple',
            '--skip-errors'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text(tei_auto_root, '//docTitle/titlePart') == TEXT_1

    @pytest.mark.parametrize(
        'relative_failed_output_path', ['', 'tei-error']
    )
    @pytest.mark.parametrize(
        'actual_abstract,expected_match', [
            (ABSTRACT_1, True),
            (NOT_MATCHING_ABSTRACT_1, False)
        ]
    )
    def test_should_filter_out_xml_if_selected_fields_are_not_matching(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper,
            actual_abstract: str,
            expected_match: bool,
            relative_failed_output_path: str,
            temp_dir: Path):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_header_tei_node([
                E.note(TITLE_1), E.lb(),
                E.note(ABSTRACT_PREFIX_1, E.lb(), ABSTRACT_1)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            title=TITLE_1,
            abstract_node=E.abstract(E.p(actual_abstract))
        )))
        failed_output_path = (
            temp_dir / relative_failed_output_path
            if relative_failed_output_path
            else ''
        )
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join(['title', 'author', 'author_aff', 'abstract']),
            'require-matching-fields': ','.join(['abstract']),
            'failed-output-path': failed_output_path,
            'matcher': 'simple'
        }), save_main_session=False)

        if not expected_match:
            assert not test_helper.tei_auto_file_path.exists()
            if failed_output_path:
                assert (failed_output_path / test_helper.tei_auto_file_path.name).exists()
        else:
            assert test_helper.tei_auto_file_path.exists()
