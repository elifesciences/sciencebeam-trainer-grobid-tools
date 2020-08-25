# pylint: disable=singleton-comparison
import logging
import re

import pytest

from sciencebeam_utils.utils.collection import flatten

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken,
    B_TAG_PREFIX,
    I_TAG_PREFIX,
    add_tag_prefix,
    strip_tag_prefix
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_trainer_grobid_tools.structured_document.simple_matching_annotator import (
    SimpleTagConfig,
    SimpleMatchingAnnotator,
    get_extended_line_token_tags,
    get_simple_tag_config_map,
    select_index_ranges,
    DEFAULT_MERGE_ENABLED,
    DEFAULT_EXTEND_TO_LINE_ENABLED
)

from tests.test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)

TAG1 = 'tag1'
TAG2 = 'tag2'

B_TAG1 = add_tag_prefix(TAG1, prefix=B_TAG_PREFIX)
I_TAG1 = add_tag_prefix(TAG1, prefix=I_TAG_PREFIX)

B_TAG2 = add_tag_prefix(TAG2, prefix=B_TAG_PREFIX)
I_TAG2 = add_tag_prefix(TAG2, prefix=I_TAG_PREFIX)


def _get_tags_of_tokens(tokens, **kwargs):
    return [t.get_tag(**kwargs) for t in tokens]


def _get_tag_values_of_tokens(tokens, **kwargs):
    return [strip_tag_prefix(tag) for tag in _get_tags_of_tokens(tokens, **kwargs)]


def _get_sub_tag_values_of_tokens(tokens):
    return _get_tag_values_of_tokens(tokens, level=2)


def _tokens_for_text(text):
    return [SimpleToken(s) for s in re.split(r'(\W)', text) if s.strip()]


def _lines_for_tokens(tokens_by_line):
    return [SimpleLine(tokens) for tokens in tokens_by_line]


def _document_for_tokens(tokens_by_line):
    return SimpleStructuredDocument(lines=_lines_for_tokens(tokens_by_line))


def _get_document_token_tags(doc: SimpleStructuredDocument):
    return [
        [
            (doc.get_text(token), doc.get_tag(token))
            for token in doc.get_tokens_of_line(line)
        ]
        for page in doc.get_pages()
        for line in doc.get_lines_of_page(page)
    ]


class TestSelectIndexRanges:
    def test_should_select_no_index_ranges(self):
        selected, unselected = select_index_ranges([])
        assert selected == []
        assert unselected == []

    def test_should_select_single_index_range(self):
        selected, unselected = select_index_ranges([
            (1, 3)
        ])
        assert selected == [(1, 3)]
        assert unselected == []

    def test_should_select_two_consequitive_index_ranges(self):
        selected, unselected = select_index_ranges([
            (1, 3), (3, 5)
        ])
        assert selected == [(1, 3), (3, 5)]
        assert unselected == []

    def test_should_select_first_longer_of_two_apart_index_ranges(self):
        selected, unselected = select_index_ranges([
            (1, 3), (103, 105)
        ])
        assert selected == [(1, 3)]
        assert unselected == [(103, 105)]

    def test_should_select_second_longer_of_two_apart_index_ranges(self):
        selected, unselected = select_index_ranges([
            (1, 3), (103, 109)
        ])
        assert selected == [(103, 109)]
        assert unselected == [(1, 3)]

    def test_should_select_two_close_and_unselect_apart_index_ranges(self):
        selected, unselected = select_index_ranges([
            (1, 3), (3, 5), (103, 105)
        ])
        assert selected == [(1, 3), (3, 5)]
        assert unselected == [(103, 105)]


class TestGetExtendedLineTokenTags:
    def test_should_fill_begining_of_line(self):
        assert get_extended_line_token_tags(
            [None, TAG1, TAG1],
            extend_to_line_enabled_map={TAG1: True},
        ) == [TAG1] * 3

    def test_should_fill_begining_of_line_with_begin_prefix(self):
        assert get_extended_line_token_tags(
            [None, B_TAG1, I_TAG1],
            extend_to_line_enabled_map={TAG1: True},
            merge_enabled_map={TAG1: False}
        ) == [B_TAG1, I_TAG1, I_TAG1]

    def test_should_fill_multi_token_begining_of_line_with_begin_prefix(self):
        assert get_extended_line_token_tags(
            [None, None, B_TAG1, I_TAG1, I_TAG1, I_TAG1],
            extend_to_line_enabled_map={TAG1: True},
            merge_enabled_map={TAG1: False}
        ) == [B_TAG1, I_TAG1, I_TAG1, I_TAG1, I_TAG1, I_TAG1]

    def test_should_fill_end_of_line(self):
        assert get_extended_line_token_tags(
            [TAG1, TAG1, None],
            extend_to_line_enabled_map={TAG1: True},
        ) == [TAG1] * 3

    def test_should_fill_end_of_line_with_begin_prefix(self):
        assert get_extended_line_token_tags(
            [B_TAG1, I_TAG1, None],
            extend_to_line_enabled_map={TAG1: True}
        ) == [B_TAG1, I_TAG1, I_TAG1]

    def test_should_fill_gaps_if_same_tag(self):
        assert get_extended_line_token_tags(
            [TAG1, None, TAG1],
            extend_to_line_enabled_map={TAG1: True}
        ) == [TAG1, TAG1, TAG1]

    def test_should_fill_gaps_if_same_tag_with_begin_prefix_and_merge_enabled(self):
        assert get_extended_line_token_tags(
            [B_TAG1, None, B_TAG1],
            extend_to_line_enabled_map={TAG1: True},
            merge_enabled_map={TAG1: True}
        ) == [B_TAG1, I_TAG1, I_TAG1]

    def test_should_adjust_begin_inside_tag_prefix_if_merge_enabled(self):
        assert get_extended_line_token_tags(
            [B_TAG1, I_TAG1, B_TAG1],
            extend_to_line_enabled_map={TAG1: True},
            merge_enabled_map={TAG1: True}
        ) == [B_TAG1, I_TAG1, I_TAG1]

    def test_should_not_fill_gaps_if_same_tag_with_begin_prefix_but_merge_disabled(self):
        assert get_extended_line_token_tags(
            [B_TAG1, None, B_TAG1],
            extend_to_line_enabled_map={TAG1: True},
            merge_enabled_map={TAG1: False}
        ) == [B_TAG1, None, B_TAG1]

    def test_should_not_fill_gaps_if_not_same_tag(self):
        assert get_extended_line_token_tags(
            [TAG1, None, TAG2],
            extend_to_line_enabled_map={TAG1: True, TAG2: True}
        ) == [TAG1, None, TAG2]

    def test_should_not_fill_line_if_minority_tag(self):
        token_tags = [None, None, TAG1, None, None]
        assert get_extended_line_token_tags(
            token_tags,
            extend_to_line_enabled_map={TAG1: True}
        ) == token_tags

    def test_should_fill_begining_of_line_if_not_enabled_by_tag_config(self):
        assert get_extended_line_token_tags(
            [None, TAG1, TAG1],
            extend_to_line_enabled_map={TAG1: False}
        ) == [None, TAG1, TAG1]

    def test_should_fill_begining_of_line_if_not_enabled_by_tag_config_with_begin_prefix(self):
        assert get_extended_line_token_tags(
            [None, B_TAG1, I_TAG1],
            extend_to_line_enabled_map={TAG1: False}
        ) == [None, B_TAG1, I_TAG1]


class TestSimpleMatchingAnnotator:
    def test_should_not_fail_on_empty_document(self):
        doc = SimpleStructuredDocument(lines=[])
        SimpleMatchingAnnotator([]).annotate(doc)

    def test_should_return_document(self):
        doc = SimpleStructuredDocument(lines=[])
        assert SimpleMatchingAnnotator([]).annotate(doc) == doc

    @pytest.mark.skip(reason='no longer throwing exception')
    def test_should_fail_with_unsupported_annotation_attribute_match_multiple(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, match_multiple=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    @pytest.mark.skip(reason='no longer throwing exception')
    def test_should_fail_with_unsupported_annotation_attribute_bonding(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, bonding=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    @pytest.mark.skip(reason='no longer throwing exception')
    def test_should_fail_with_unsupported_annotation_attribute_require_next(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, require_next=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    @pytest.mark.skip(reason='no longer throwing exception')
    def test_should_fail_with_unsupported_annotation_attribute_sub_annotations(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, sub_annotations=[TargetAnnotation('sub', TAG2)])
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    def test_should_not_fail_on_empty_line_with_blank_token(self):
        target_annotations = [
            TargetAnnotation('this is. matching', TAG1)
        ]
        doc = _document_for_tokens([[SimpleToken('')]])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)

    def test_should_annotate_exactly_matching(self):
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is matching', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_match_case_insensitive(self):
        matching_tokens = _tokens_for_text('This Is Matching')
        target_annotations = [
            TargetAnnotation('tHIS iS mATCHING', TAG1)
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(matching_tokens)])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_match_single_quotes_with_double_quotes(self):
        matching_tokens = _tokens_for_text('"this is matching"')
        target_annotations = [
            TargetAnnotation('\'this is matching\'', TAG1)
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(matching_tokens)])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_match_apos_with_double_quotes(self):
        matching_tokens = _tokens_for_text('"this is matching"')
        target_annotations = [
            TargetAnnotation('&apos;this is matching&apos;', TAG1)
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(matching_tokens)])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_prefer_word_boundaries(self):
        pre_tokens = _tokens_for_text('this')
        matching_tokens = _tokens_for_text('is')
        post_tokens = _tokens_for_text('miss')
        target_annotations = [
            TargetAnnotation('is', TAG1)
        ]
        doc = _document_for_tokens([
            pre_tokens + matching_tokens + post_tokens
        ])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    def test_should_annotate_fuzzily_matching(self):
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is. matching', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_using_alternative_spellings(self):
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('alternative spelling', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={
                TAG1: SimpleTagConfig(alternative_spellings={
                    'alternative spelling': ['this is matching']
                })
            }
        ).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_space_after_dot_short_sequence(self):
        matching_tokens = [
            SimpleToken('A.B.,')
        ]
        target_annotations = [
            TargetAnnotation('A. B.', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_comma_after_short_sequence(self):
        matching_tokens = [
            SimpleToken('Name,'),
        ]
        target_annotations = [
            TargetAnnotation('Name', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_including_final_dot(self):
        matching_tokens = _tokens_for_text('this is matching.')
        target_annotations = [
            TargetAnnotation('this is matching.', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_dots_after_capitals_in_target_annotation(self):
        matching_tokens = _tokens_for_text('PO Box 12345')
        target_annotations = [
            TargetAnnotation('P.O. Box 12345', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    @pytest.mark.skip(reason="this is currently failing, needs more investigation")
    def test_should_annotate_ignoring_dots_after_capitals_in_document(self):
        matching_tokens = _tokens_for_text('P.O. Box 12345')
        target_annotations = [
            TargetAnnotation('PO Box 12345', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_not_annotate_with_local_matching(self):
        tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is matching but not fully matching', TAG1)
        ]
        doc = _document_for_tokens([tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(tokens) == [None] * len(tokens)

    def test_should_annotate_author_aff_preceding_number(self):
        number_tokens = _tokens_for_text('1')
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is matching', TAG1)
        ]
        doc = _document_for_tokens([number_tokens, matching_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={TAG1: SimpleTagConfig(match_prefix_regex=r'(?=^|\n)\d\s*$')}
        ).annotate(doc)
        assert _get_tag_values_of_tokens(number_tokens) == [TAG1] * len(number_tokens)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_not_annotate_author_aff_preceding_number_if_it_is_following_text(self):
        number_tokens = _tokens_for_text('Smith 1')
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is matching', TAG1)
        ]
        doc = _document_for_tokens([number_tokens, matching_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={TAG1: SimpleTagConfig(match_prefix_regex=r'(?=^|\n)\d\s*$')}
        ).annotate(doc)
        assert _get_tag_values_of_tokens(number_tokens) == [None] * len(number_tokens)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    @log_on_exception
    def test_should_not_annotate_author_aff_label_between_author_names(self):
        author_tokens = _tokens_for_text('Mary 1 , Smith 1')
        aff_tokens = _tokens_for_text('University of Science')
        target_annotations = [
            TargetAnnotation(['Mary', 'Smith'], TAG1),
            TargetAnnotation(['1', 'University of Science'], TAG2)
        ]
        doc = _document_for_tokens([author_tokens, aff_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={TAG1: SimpleTagConfig(extend_to_line_enabled=True)}
        ).annotate(doc)
        assert _get_tag_values_of_tokens(author_tokens) == [TAG1] * len(author_tokens)
        assert _get_tag_values_of_tokens(aff_tokens) == [TAG2] * len(aff_tokens)

    @log_on_exception
    def test_should_annotate_separate_author_aff_with_begin_prefix(self):
        aff1_tokens = _tokens_for_text('University of Science')
        aff2_tokens = _tokens_for_text('University of Madness')
        target_annotations = [
            TargetAnnotation(['1', 'University of Science'], TAG1),
            TargetAnnotation(['2', 'University of Madness'], TAG1)
        ]
        doc = _document_for_tokens([aff1_tokens, aff2_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={}
        ).annotate(doc)
        assert (
            _get_tags_of_tokens(aff1_tokens) == [B_TAG1] + [I_TAG1] * (len(aff1_tokens) - 1)
        )
        assert (
            _get_tags_of_tokens(aff2_tokens) == [B_TAG1] + [I_TAG1] * (len(aff2_tokens) - 1)
        )

    def test_should_annotate_abstract_section_heading(self):
        matching_tokens = _tokens_for_text('Abstract\nthis is matching.')
        target_annotations = [
            TargetAnnotation('this is matching.', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={TAG1: SimpleTagConfig(match_prefix_regex=r'(abstract|summary)\s*$')}
        ).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_not_annotate_fuzzily_matching_with_many_differences(self):
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('txhxixsx ixsx mxaxtxcxhxixnxgx', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [None] * len(matching_tokens)

    def test_should_not_annotate_not_matching(self):
        not_matching_tokens = _tokens_for_text('something completely different')
        target_annotations = [
            TargetAnnotation('this is matching', TAG1)
        ]
        doc = _document_for_tokens([not_matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(not_matching_tokens) == [None] * len(not_matching_tokens)

    def test_should_annotate_exactly_matching_across_multiple_lines(self):
        matching_tokens_per_line = [
            _tokens_for_text('this is matching'),
            _tokens_for_text('and continues here')
        ]
        matching_tokens = flatten(matching_tokens_per_line)
        target_annotations = [
            TargetAnnotation('this is matching and continues here', TAG1)
        ]
        doc = _document_for_tokens(matching_tokens_per_line)
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_over_multiple_lines_with_tag_transition(self):
        tag1_tokens_by_line = [
            _tokens_for_text('this may'),
            _tokens_for_text('match')
        ]
        tag1_tokens = flatten(tag1_tokens_by_line)
        tag2_tokens_by_line = [
            _tokens_for_text('another'),
            _tokens_for_text('tag here')
        ]
        tag2_tokens = flatten(tag2_tokens_by_line)
        tokens_by_line = [
            tag1_tokens_by_line[0],
            tag1_tokens_by_line[1] + tag2_tokens_by_line[0],
            tag2_tokens_by_line[1]
        ]
        target_annotations = [
            TargetAnnotation('this may match', TAG1),
            TargetAnnotation('another tag here', TAG2)
        ]
        doc = _document_for_tokens(tokens_by_line)
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(tag1_tokens) == [TAG1] * len(tag1_tokens)
        assert _get_tag_values_of_tokens(tag2_tokens) == [TAG2] * len(tag2_tokens)

    def test_should_annotate_multiple_value_annotation(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens = _tokens_for_text('john smith')
        post_tokens = _tokens_for_text('the author')
        doc_tokens = pre_tokens + matching_tokens + post_tokens
        target_annotations = [
            TargetAnnotation(['john', 'smith'], TAG1)
        ]
        doc = _document_for_tokens([doc_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    def test_should_annotate_multiple_value_annotation_in_reverse_order(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens = _tokens_for_text('john smith')
        post_tokens = _tokens_for_text('the author')
        doc_tokens = pre_tokens + matching_tokens + post_tokens
        target_annotations = [
            TargetAnnotation(['smith', 'john'], TAG1)
        ]
        doc = _document_for_tokens([doc_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    def test_should_not_annotate_multiple_value_annotation_too_far_away(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens = _tokens_for_text('smith')
        post_tokens = _tokens_for_text('etc') * 40 + _tokens_for_text('john')
        doc_tokens = pre_tokens + matching_tokens + post_tokens
        target_annotations = [
            TargetAnnotation(['john', 'smith'], TAG1)
        ]
        doc = _document_for_tokens([doc_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        LOGGER.debug('doc: %s', _get_document_token_tags(doc))
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    @log_on_exception
    def test_should_annotate_and_merge_multiple_authors_annotation(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens = _tokens_for_text('john smith, mary maison')
        post_tokens = _tokens_for_text('the author')
        target_annotations = [
            TargetAnnotation(['john', 'smith'], TAG1),
            TargetAnnotation(['mary', 'maison'], TAG1)
        ]
        doc = _document_for_tokens([pre_tokens, matching_tokens, post_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={
                TAG1: SimpleTagConfig(extend_to_line_enabled=True, merge_enabled=True)
            }
        ).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    @log_on_exception
    def test_should_annotate_but_not_merge_multiple_authors_annotation_too_far_apart(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens_1 = _tokens_for_text('john smith')
        mid_tokens = _tokens_for_text('etc') * 5
        matching_tokens_2 = _tokens_for_text('mary maison')
        post_tokens = _tokens_for_text('the author')
        target_annotations = [
            TargetAnnotation(['john', 'smith'], TAG1),
            TargetAnnotation(['mary', 'maison'], TAG1)
        ]
        doc = _document_for_tokens([
            pre_tokens, matching_tokens_1, mid_tokens, matching_tokens_2, post_tokens
        ])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens_1) == [TAG1] * len(matching_tokens_1)
        assert _get_tag_values_of_tokens(matching_tokens_2) == [TAG1] * len(matching_tokens_2)
        assert _get_tag_values_of_tokens(mid_tokens) == [None] * len(mid_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    @log_on_exception
    def test_should_annotate_whole_line(self):
        pre_tokens = _tokens_for_text('this is')
        matching_tokens = _tokens_for_text('john smith 1, mary maison 2')
        post_tokens = _tokens_for_text('the author')
        target_annotations = [
            TargetAnnotation(['john', 'smith'], TAG1),
            TargetAnnotation(['mary', 'maison'], TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(
            target_annotations,
            tag_config_map={TAG1: SimpleTagConfig(extend_to_line_enabled=True)}
        ).annotate(doc)
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tag_values_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tag_values_of_tokens(post_tokens) == [None] * len(post_tokens)

    def test_should_annotate_references(self):
        matching_tokens_list = [
            _tokens_for_text('1 this is reference A'),
            _tokens_for_text('2 this is reference B'),
            _tokens_for_text('3 this is reference C')
        ]
        matching_tokens = flatten(matching_tokens_list)
        target_annotations = [
            TargetAnnotation('this is reference A', TAG1),
            TargetAnnotation('this is reference B', TAG1),
            TargetAnnotation('this is reference C', TAG1)
        ]
        pre_tokens = [_tokens_for_text('previous line')] * 5
        doc = _document_for_tokens(pre_tokens + matching_tokens_list)
        SimpleMatchingAnnotator(
            target_annotations,
            lookahead_sequence_count=3
        ).annotate(doc)
        LOGGER.debug('doc: %s', _get_document_token_tags(doc))
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_references_with_sub_tag(self):
        matching_tokens_list = [
            _tokens_for_text('1 this is reference A')
        ]
        matching_tokens = flatten(matching_tokens_list)
        target_annotations = [
            TargetAnnotation('1 this is reference A', TAG1, sub_annotations=[
                TargetAnnotation('1', TAG2)
            ]),
        ]
        pre_tokens = [_tokens_for_text('previous line')] * 5
        doc = _document_for_tokens(pre_tokens + matching_tokens_list)
        SimpleMatchingAnnotator(
            target_annotations,
            lookahead_sequence_count=3,
            extend_to_line_enabled=False,
            use_sub_annotations=True
        ).annotate(doc)
        LOGGER.debug('doc: %s', _get_document_token_tags(doc))
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_sub_tag_values_of_tokens(matching_tokens) == (
            [TAG2] + [None] * (len(matching_tokens) - 1)
        )

    def test_should_annotate_references_with_sub_tag_with_extend_to_line(self):
        matching_tokens_list = [
            _tokens_for_text('1 this is reference A')
        ]
        matching_tokens = flatten(matching_tokens_list)
        target_annotations = [
            TargetAnnotation('1 this is reference A', TAG1, sub_annotations=[
                TargetAnnotation('1', TAG2)
            ]),
        ]
        pre_tokens = [_tokens_for_text('previous line')] * 5
        doc = _document_for_tokens(pre_tokens + matching_tokens_list)
        SimpleMatchingAnnotator(
            target_annotations,
            lookahead_sequence_count=3,
            extend_to_line_enabled=True,
            use_sub_annotations=True
        ).annotate(doc)
        LOGGER.debug('doc: %s', _get_document_token_tags(doc))
        assert _get_tag_values_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_sub_tag_values_of_tokens(matching_tokens) == (
            [TAG2] + [None] * (len(matching_tokens) - 1)
        )


class TestGetSimpleTagConfigMap:
    def test_should_parse_merge_flag(self):
        tag_config_map = get_simple_tag_config_map({
            'any': {
                'tag1': 'xpath1',
                'tag1.merge': 'false',
                'tag2': 'xpath2',
                'tag2.merge': 'true',
                'tag3': 'xpath3'
            }
        })
        assert set(tag_config_map.keys()) == {'tag1', 'tag2', 'tag3'}
        assert tag_config_map['tag1'].merge_enabled is False
        assert tag_config_map['tag2'].merge_enabled is True
        assert tag_config_map['tag3'].merge_enabled is DEFAULT_MERGE_ENABLED

    def test_should_parse_extend_to_line_flag(self):
        tag_config_map = get_simple_tag_config_map({
            'any': {
                'tag1': 'xpath1',
                'tag1.extend-to-line': 'false',
                'tag2': 'xpath2',
                'tag2.extend-to-line': 'true',
                'tag3': 'xpath3'
            }
        })
        assert set(tag_config_map.keys()) == {'tag1', 'tag2', 'tag3'}
        assert tag_config_map['tag1'].extend_to_line_enabled is False
        assert tag_config_map['tag2'].extend_to_line_enabled is True
        assert tag_config_map['tag3'].extend_to_line_enabled is DEFAULT_EXTEND_TO_LINE_ENABLED

    def test_should_parse_match_prefix_regex(self):
        tag_config_map = get_simple_tag_config_map({
            'any': {
                'tag1': 'xpath1',
                'tag1.match-prefix-regex': 'regex1'
            }
        })
        assert set(tag_config_map.keys()) == {'tag1'}
        tag_config = tag_config_map['tag1']
        assert tag_config.match_prefix_regex == 'regex1'

    def test_should_parse_alternative_spellings(self):
        tag_config_map = get_simple_tag_config_map({
            'any': {
                'tag1': 'xpath1',
                'tag1.alternative-spellings': '\n Key 1=Alternative 1,Alternative 2\n'
            }
        })
        assert set(tag_config_map.keys()) == {'tag1'}
        tag_config = tag_config_map['tag1']
        assert tag_config.alternative_spellings == {
            'Key 1': ['Alternative 1', 'Alternative 2']
        }

    def test_should_parse_block(self):
        tag_config_map = get_simple_tag_config_map({
            'any': {
                'tag1': 'xpath1',
                'tag1.block': 'block1',
                'tag2': 'xpath2'
            }
        })
        assert set(tag_config_map.keys()) == {'tag1', 'tag2'}
        assert tag_config_map['tag1'].block_name == 'block1'
        assert tag_config_map['tag2'].block_name is None
