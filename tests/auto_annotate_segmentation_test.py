import logging
from pathlib import Path
from typing import List, Union

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.utils.xml import get_xpath_text_list

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    MatcherNames
)

from sciencebeam_trainer_grobid_tools.auto_annotate_segmentation import (
    main
)

from .test_utils import log_on_exception, dict_to_args

from .auto_annotate_test_utils import (
    get_target_xml_node,
    get_nodes_text,
    get_tei_nodes_for_text,
    get_default_target_xml_node,
    SingleFileAutoAnnotateEndToEndTestHelper
)


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.segmentation.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).segmentation.tei.xml/\1.xml/'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'

LABEL_1 = '1'
REFERENCE_TEXT_1 = 'reference A'

TITLE_1 = 'Chocolate bars for mice'
ABSTRACT_PREFIX_1 = 'Abstract'
ABSTRACT_1 = (
    'This study explores the nutritious value of chocolate bars for mice.'
)
NOT_MATCHING_ABSTRACT_1 = (
    'Something different.'
)

FIGURE_LABEL_1 = 'Figure 1'
FIGURE_LABEL_2 = 'Figure 2'

SECTION_TITLE_1 = 'Section Title 1'
SECTION_TITLE_2 = 'Section Title 2'

TEXT_1 = 'text 1'
TEXT_2 = 'text 2'


def get_segmentation_tei_node(
        text_items: List[Union[etree.Element, str]]) -> etree.Element:
    return E.tei(E.text(*text_items))


def get_training_tei_node(
        items: List[Union[etree.Element, str]]) -> etree.Element:
    return get_segmentation_tei_node(items)


def get_default_tei_node() -> etree.Element:
    return get_segmentation_tei_node([E.note(TOKEN_1)])


def get_jats_reference_node(label: str, text: str) -> etree.Element:
    ref = E.ref()
    if label:
        ref.append(E.label(label))
    ref.append(E('mixed-citation', text))
    return ref


@pytest.fixture(name='test_helper')
def _test_helper(temp_dir: Path) -> SingleFileAutoAnnotateEndToEndTestHelper:
    return SingleFileAutoAnnotateEndToEndTestHelper(
        temp_dir=temp_dir,
        tei_filename=TEI_FILENAME_1,
        tei_filename_regex=TEI_FILENAME_REGEX
    )


@log_on_exception
class TestEndToEnd(object):
    def test_should_auto_annotate_title_as_front(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.note(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/front') == [TOKEN_1]

    def test_should_auto_annotate_using_simple_matcher(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.note(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TOKEN_1)
        ))
        main([
            *test_helper.main_args,
            '--matcher=simple'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/front') == [TOKEN_1]

    def test_should_merge_front_tags_and_include_preceeding_text(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        front_tei_text = '\n'.join([
            'Before',
            TITLE_1,
            'Other',
            ABSTRACT_1
        ])
        tei_text = '\n'.join([
            front_tei_text,
            'After'
        ])
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.write_xml_root(
            get_target_xml_node(
                title=TITLE_1,
                abstract_node=E.abstract(E.p(ABSTRACT_1))
            )
        )
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'title',
                'abstract'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//front') == [front_tei_text]

    def test_should_auto_annotate_body_and_back_section(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_1),
                '\n',
                E.p(TEXT_1)
            )
        ]
        target_back_content_nodes = [
            E.sec(
                E.title(SECTION_TITLE_2),
                '\n',
                E.p(TEXT_2)
            )
        ]
        body_tei_text = get_nodes_text(target_body_content_nodes)
        back_tei_text = get_nodes_text(target_back_content_nodes)
        tei_text = '\n'.join([body_tei_text, back_tei_text])
        LOGGER.debug('tei_text: %s', tei_text)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(
                body_nodes=target_body_content_nodes,
                back_nodes=target_back_content_nodes
            )
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'body_section_title',
                'body_section_paragraph',
                'back_section_title',
                'back_section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//body') == [body_tei_text]
        assert get_xpath_text_list(tei_auto_root, '//div[@type="annex"]') == [back_tei_text]

    def test_should_auto_annotate_body_and_back_top_level_section_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [E.p(TEXT_1)]
        target_back_content_nodes = [E.p(TEXT_2)]
        tei_text = '\n'.join([
            get_nodes_text(target_body_content_nodes),
            get_nodes_text(target_back_content_nodes)
        ])
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(
                body_nodes=target_body_content_nodes,
                back_nodes=target_back_content_nodes
            )
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'body_section_paragraph',
                'back_section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//body') == [TEXT_1]
        assert get_xpath_text_list(tei_auto_root, '//div[@type="annex"]') == [TEXT_2]

    def test_should_auto_annotate_body_and_back_list_item_section_paragraphs(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [E.list(E('list-item', E.p(TEXT_1)))]
        target_back_content_nodes = [E.list(E('list-item', E.p(TEXT_2)))]
        tei_text = '\n'.join([
            get_nodes_text(target_body_content_nodes),
            get_nodes_text(target_back_content_nodes)
        ])
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(
                body_nodes=target_body_content_nodes,
                back_nodes=target_back_content_nodes
            )
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'body_section_paragraph',
                'back_section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//body') == [TEXT_1]
        assert get_xpath_text_list(tei_auto_root, '//div[@type="annex"]') == [TEXT_2]

    def test_should_auto_annotate_acknowledgment_section_as_acknowledgment(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E.ack(
                E.title(SECTION_TITLE_2),
                '\n',
                E.p(TEXT_2)
            )
        ]
        tei_text = get_nodes_text(target_back_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'body_section_title',
                'body_section_paragraph',
                'back_section_title',
                'back_section_paragraph',
                'acknowledgment_section_title',
                'acknowledgment_section_paragraph'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//div[@type="acknowledgment"]') == [tei_text]

    def test_should_auto_annotate_appendix_section_as_annex(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_back_content_nodes = [
            E('app-group', *[
                E.title('Appendix'),
                '\n',
                E.app(
                    E.title(SECTION_TITLE_2),
                    '\n',
                    E.p(TEXT_2)
                )
            ])
        ]
        tei_text = get_nodes_text(target_back_content_nodes)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(back_nodes=target_back_content_nodes)
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'body_section_title',
                'body_section_paragraph',
                'back_section_title',
                'back_section_paragraph',
                'appendix_group_title',
                'appendix'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//div[@type="annex"]') == [tei_text]

    def test_should_auto_annotate_body_and_back_figure_label_title_caption_as_body(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E.fig(
                E.label(FIGURE_LABEL_1),
                ' ',
                E.caption(
                    E.title(SECTION_TITLE_1),
                    '\n',
                    E.p(TEXT_1)
                )
            )
        ]
        target_back_content_nodes = [
            E.fig(
                E.label(FIGURE_LABEL_2),
                ' ',
                E.caption(
                    E.title(SECTION_TITLE_2),
                    '\n',
                    E.p(TEXT_2)
                )
            )
        ]
        body_tei_text = get_nodes_text(target_body_content_nodes)
        back_tei_text = get_nodes_text(target_back_content_nodes)
        tei_text = '\n'.join([body_tei_text, back_tei_text])
        LOGGER.debug('tei_text: %s', tei_text)
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(
                body_nodes=target_body_content_nodes,
                back_nodes=target_back_content_nodes
            )
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'figure'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//body') == [tei_text]

    def test_should_auto_annotate_body_and_back_table_label_title_caption_as_body(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        target_body_content_nodes = [
            E('table-wrap', *[
                E.label(FIGURE_LABEL_1),
                ' ',
                E.caption(
                    E.title(SECTION_TITLE_1),
                    '\n',
                    E.p(TEXT_1)
                )
            ])
        ]
        target_back_content_nodes = [
            E('table-wrap', *[
                E.label(FIGURE_LABEL_2),
                ' ',
                E.caption(
                    E.title(SECTION_TITLE_2),
                    '\n',
                    E.p(TEXT_2)
                )
            ])
        ]
        body_tei_text = get_nodes_text(target_body_content_nodes)
        back_tei_text = get_nodes_text(target_back_content_nodes)
        tei_text = '\n'.join([body_tei_text, back_tei_text])
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_training_tei_node(get_tei_nodes_for_text(tei_text))
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(
                body_nodes=target_body_content_nodes,
                back_nodes=target_back_content_nodes
            )
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'fields': ','.join([
                'table'
            ])
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//body') == [tei_text]

    def test_should_process_specific_file(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'source-base-path': None,
            'source-path': str(test_helper.tei_raw_file_path)
        }), save_main_session=False)

        assert test_helper.get_tei_auto_root() is not None

    def test_should_skip_existing_output_file_if_resume_is_enabled(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        test_helper.tei_auto_file_path.parent.mkdir()
        test_helper.tei_auto_file_path.write_bytes(b'existing')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'resume': True
        }), save_main_session=False)

        assert test_helper.tei_auto_file_path.read_bytes() == b'existing'

    def test_should_run_locally_without_beam_if_workers_more_than_one(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        test_helper.tei_auto_file_path.parent.mkdir()
        test_helper.tei_auto_file_path.write_bytes(b'existing')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'num_workers': 2
        }), save_main_session=False)
        assert test_helper.get_tei_auto_root() is not None

    def test_should_run_locally_using_multi_processing(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        test_helper.tei_auto_file_path.parent.mkdir()
        test_helper.tei_auto_file_path.write_bytes(b'existing')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'num_workers': 2,
            'multi-processing': True
        }), save_main_session=False)
        assert test_helper.get_tei_auto_root() is not None

    def test_should_write_debug_match(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper, temp_dir: Path):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(get_default_tei_node()))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_default_target_xml_node()))
        debug_match_path = temp_dir.joinpath('debug.csv')
        main(dict_to_args({
            **test_helper.main_args_dict,
            'matcher': MatcherNames.COMPLEX,
            'debug-match': str(debug_match_path)
        }), save_main_session=False)

        assert debug_match_path.exists()

    def test_should_preserve_existing_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.page(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            E.article(E.front(
            ))
        ))
        main([
            *test_helper.main_args
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/page') == [TOKEN_1]

    def test_should_always_preserve_specified_existing_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TOKEN_1),
                E.lb(),
                E.page(TOKEN_2),
                E.lb(),
                E.note(TOKEN_3),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=' '.join([TOKEN_1, TOKEN_2, TOKEN_3]))
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'always-preserve-fields': 'page'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/page') == [TOKEN_2]

    def test_should_always_preserve_specified_existing_tag_when_merging_front(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TITLE_1),
                E.lb(),
                E.page(TOKEN_2),
                E.lb(),
                E.note(ABSTRACT_1),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=TITLE_1, abstract_node=E.abstract(E.p(ABSTRACT_1)))
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'always-preserve-fields': 'page'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/page') == [TOKEN_2]
        assert get_xpath_text_list(tei_auto_root, '//text/front') == [TITLE_1, ABSTRACT_1]

    def test_should_always_preserve_reference_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        _common_tokens = [TOKEN_2, TOKEN_3]
        _reference_text = ' '.join(_common_tokens) + ' this is a reference 1'
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TOKEN_1),
                E.lb(),
                E.listBibl(_reference_text),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(title=' '.join([TOKEN_1] + _common_tokens))
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'no-preserve-tags': True,
            'always-preserve-fields': 'reference'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/listBibl') == [_reference_text]

    def test_should_auto_annotate_reference_without_separate_label_tag(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TOKEN_1),
                E.lb(),
                E.note(LABEL_1 + ' ' + REFERENCE_TEXT_1),
                E.lb()
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            get_target_xml_node(reference_nodes=[
                get_jats_reference_node(LABEL_1, REFERENCE_TEXT_1),
            ])
        ))
        main(dict_to_args({
            **test_helper.main_args_dict,
            'no-preserve-tags': True,
            'fields': 'reference',
            'xml-mapping-overrides': 'reference.use-raw-text=true'
        }), save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/listBibl') == [
            LABEL_1 + ' ' + REFERENCE_TEXT_1
        ]

    def test_should_not_preserve_exclude_existing_tag_and_use_body_by_default(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([E.page(TOKEN_1)])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(
            E.article(E.front(
            ))
        ))
        main([
            *test_helper.main_args,
            '--no-preserve-fields=page'
        ], save_main_session=False)

        tei_auto_root = test_helper.get_tei_auto_root()
        assert get_xpath_text_list(tei_auto_root, '//text/page') == []
        assert get_xpath_text_list(tei_auto_root, '//text/body') == [TOKEN_1]

    @pytest.mark.parametrize(
        'relative_failed_output_path', ['tei-error', '']
    )
    @pytest.mark.parametrize(
        'actual_abstract,expected_abstract,required_fields,expected_match', [
            (ABSTRACT_1, ABSTRACT_1, '', True),
            (NOT_MATCHING_ABSTRACT_1, ABSTRACT_1, '', False),
            ('', ABSTRACT_1, '', False),
            (ABSTRACT_1, '', '', True),
            (ABSTRACT_1, '', 'abstract', False)
        ]
    )
    def test_should_filter_out_xml_if_selected_fields_are_not_matching(
            self, test_helper: SingleFileAutoAnnotateEndToEndTestHelper,
            actual_abstract: str,
            expected_abstract: str,
            expected_match: bool,
            required_fields: str,
            relative_failed_output_path: str,
            temp_dir: Path):
        test_helper.tei_raw_file_path.write_bytes(etree.tostring(
            get_segmentation_tei_node([
                E.note(TITLE_1), E.lb(),
                E.note(ABSTRACT_PREFIX_1, E.lb(), actual_abstract)
            ])
        ))
        test_helper.xml_file_path.write_bytes(etree.tostring(get_target_xml_node(
            title=TITLE_1,
            abstract_node=(
                E.abstract(E.p(expected_abstract))
                if expected_abstract
                else None
            )
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
            'required-fields': required_fields,
            'failed-output-path': failed_output_path,
            'matcher': 'simple'
        }), save_main_session=False)

        if not expected_match:
            assert not test_helper.tei_auto_file_path.exists()
            if failed_output_path:
                assert (failed_output_path / test_helper.tei_auto_file_path.name).exists()
        else:
            assert test_helper.tei_auto_file_path.exists()
