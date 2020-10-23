from sciencebeam_trainer_grobid_tools.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_trainer_grobid_tools.annotation.checks import (
    is_structured_document_passing_checks
)

from ..structured_document.simple_document_builder import SimpleDocumentBuilder


TAG_1 = 'tag1'
VALUE_1 = 'value1'


class TestIsStructuredDocumentPassingChecks:
    def test_should_return_true_without_required_matching_fields(self):
        assert is_structured_document_passing_checks(
            SimpleDocumentBuilder().doc,
            require_matching_fields=[],
            required_fields=[],
            target_annotations=[
                TargetAnnotation(name=TAG_1, value=VALUE_1)
            ]
        )

    def test_should_return_false_if_required_matching_fields_are_not_tagged(self):
        assert not is_structured_document_passing_checks(
            SimpleDocumentBuilder().doc,
            require_matching_fields=[TAG_1],
            required_fields=[],
            target_annotations=[
                TargetAnnotation(name=TAG_1, value=VALUE_1)
            ]
        )

    def test_should_return_true_if_required_matching_fields_was_tagged_and_is_matching(self):
        assert is_structured_document_passing_checks(
            SimpleDocumentBuilder().write_entity(TAG_1, VALUE_1).doc,
            require_matching_fields=[TAG_1],
            required_fields=[],
            target_annotations=[
                TargetAnnotation(name=TAG_1, value=VALUE_1)
            ]
        )

    def test_should_return_false_if_required_matching_fields_was_tagged_but_is_not_matching(self):
        assert not is_structured_document_passing_checks(
            SimpleDocumentBuilder().write_entity(TAG_1, 'other').doc,
            require_matching_fields=[TAG_1],
            required_fields=[],
            target_annotations=[
                TargetAnnotation(name=TAG_1, value=VALUE_1)
            ]
        )

    def test_should_return_true_if_not_required_fields_is_missing_from_target_values(self):
        assert is_structured_document_passing_checks(
            SimpleDocumentBuilder().write_entity(TAG_1, 'other').doc,
            require_matching_fields=[TAG_1],
            required_fields=[],
            target_annotations=[
                TargetAnnotation(name='other', value=VALUE_1)
            ]
        )

    def test_should_return_false_if_required_fields_is_missing_from_target_values(self):
        assert not is_structured_document_passing_checks(
            SimpleDocumentBuilder().write_entity(TAG_1, 'other').doc,
            require_matching_fields=[TAG_1],
            required_fields=[TAG_1],
            target_annotations=[
                TargetAnnotation(name='other', value=VALUE_1)
            ]
        )

    def test_should_implictily_select_required_fields(self):
        assert not is_structured_document_passing_checks(
            SimpleDocumentBuilder().write_entity(TAG_1, 'other').doc,
            require_matching_fields=[],
            required_fields=[TAG_1],
            target_annotations=[
                TargetAnnotation(name='other', value=VALUE_1)
            ]
        )
