import pytest

from sciencebeam_utils.utils.collection import flatten

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_trainer_grobid_tools.structured_document.simple_matching_annotator import (
    SimpleMatchingAnnotator
)


TAG1 = 'tag1'
TAG2 = 'tag2'


def _get_tags_of_tokens(tokens):
    return [t.get_tag() for t in tokens]


def _tokens_for_text(text):
    return [SimpleToken(s) for s in text.split(' ')]


def _lines_for_tokens(tokens_by_line):
    return [SimpleLine(tokens) for tokens in tokens_by_line]


def _document_for_tokens(tokens_by_line):
    return SimpleStructuredDocument(lines=_lines_for_tokens(tokens_by_line))


class TestSimpleMatchingAnnotator:
    def test_should_not_fail_on_empty_document(self):
        doc = SimpleStructuredDocument(lines=[])
        SimpleMatchingAnnotator([]).annotate(doc)

    def test_should_return_document(self):
        doc = SimpleStructuredDocument(lines=[])
        assert SimpleMatchingAnnotator([]).annotate(doc) == doc

    def test_should_fail_with_unsupported_annotation_attribute_match_multiple(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, match_multiple=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    def test_should_fail_with_unsupported_annotation_attribute_bonding(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, bonding=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

    def test_should_fail_with_unsupported_annotation_attribute_require_next(self):
        with pytest.raises(NotImplementedError):
            target_annotations = [
                TargetAnnotation('test', TAG1, require_next=True)
            ]
            doc = _document_for_tokens([_tokens_for_text('test')])
            SimpleMatchingAnnotator(target_annotations).annotate(doc)

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
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_match_case_insensitive(self):
        matching_tokens = _tokens_for_text('This Is Matching')
        target_annotations = [
            TargetAnnotation('tHIS iS mATCHING', TAG1)
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(matching_tokens)])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

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
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
        assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
        assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

    def test_should_annotate_fuzzily_matching(self):
        matching_tokens = _tokens_for_text('this is matching')
        target_annotations = [
            TargetAnnotation('this is. matching', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_space_after_dot_short_sequence(self):
        matching_tokens = [
            SimpleToken('A.B.,')
        ]
        target_annotations = [
            TargetAnnotation('A. B.', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_comma_after_short_sequence(self):
        matching_tokens = [
            SimpleToken('Name,'),
        ]
        target_annotations = [
            TargetAnnotation('Name', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

    def test_should_annotate_ignoring_dots_after_capitals_in_target_annotation(self):
        matching_tokens = _tokens_for_text('PO Box 12345')
        target_annotations = [
            TargetAnnotation('P.O. Box 12345', TAG1)
        ]
        doc = _document_for_tokens([matching_tokens])
        SimpleMatchingAnnotator(target_annotations).annotate(doc)
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

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
        assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

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
        assert _get_tags_of_tokens(tag1_tokens) == [TAG1] * len(tag1_tokens)
        assert _get_tags_of_tokens(tag2_tokens) == [TAG2] * len(tag2_tokens)
