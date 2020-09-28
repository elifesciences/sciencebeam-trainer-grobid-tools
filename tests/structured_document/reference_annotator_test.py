from sciencebeam_trainer_grobid_tools.structured_document.reference_annotator import (
    DEFAULT_IDNO_PREFIX_REGEX,
    get_prefix_extended_token_tags,
    get_suffix_extended_token_tags,
    get_etal_mapped_tags
)


class TestPrefixExtendedTokenTags:
    def test_should_extend_doi_prefix(self):
        assert get_prefix_extended_token_tags(
            [None, None, 'b-reference-doi'],
            ['DOI', ':', '12345'],
            prefix_regex_by_tag_map={'reference-doi': DEFAULT_IDNO_PREFIX_REGEX}
        ) == ['b-reference-doi', 'i-reference-doi', 'i-reference-doi']

    def test_not_should_extend_to_other_prefix_text(self):
        assert get_prefix_extended_token_tags(
            [None, None, None, None, None, None, 'b-reference-doi'],
            ['some', 'other', 'text', ',', 'DOI', ':', '12345'],
            prefix_regex_by_tag_map={'reference-doi': DEFAULT_IDNO_PREFIX_REGEX}
        ) == [
            None, None, None, None,
            'b-reference-doi', 'i-reference-doi', 'i-reference-doi'
        ]

    def test_should_not_extend_other_tag(self):
        assert get_prefix_extended_token_tags(
            [None, None, 'b-other'],
            ['DOI', ':', '12345'],
            prefix_regex_by_tag_map={'reference-doi': DEFAULT_IDNO_PREFIX_REGEX}
        ) == [None, None, 'b-other']


class TestSuffixExtendedTokenTags:
    def test_should_extend_dot_after_author_initials(self):
        assert get_suffix_extended_token_tags(
            ['b-reference-author', 'i-reference-author', None, None],
            ['Smith', ', J', '.', 'other'],
            enabled_tags={'reference-author'}
        ) == ['b-reference-author', 'i-reference-author', 'i-reference-author', None]


class TestGetEtalMappedTags:
    def test_should_map_etal_after_author_to_author(self):
        assert get_etal_mapped_tags([
            'b-reference-author', 'i-reference-author',
            'b-reference-etal', 'i-reference-etal'
        ], etal_sub_tag='reference-etal', etal_merge_enabled_sub_tags={
            'reference-author', 'reference-editor'
        }) == [
            'b-reference-author', 'i-reference-author',
            'b-reference-author', 'i-reference-author'
        ]

    def test_should_map_etal_after_editor_to_editor(self):
        assert get_etal_mapped_tags([
            'b-reference-editor', 'i-reference-editor',
            'b-reference-etal', 'i-reference-etal'
        ], etal_sub_tag='reference-etal', etal_merge_enabled_sub_tags={
            'reference-author', 'reference-editor'
        }) == [
            'b-reference-editor', 'i-reference-editor',
            'b-reference-editor', 'i-reference-editor'
        ]

    def test_should_map_etal_after_author_tag_followed_by_none_tag(self):
        assert get_etal_mapped_tags([
            'b-reference-author', 'i-reference-author',
            None,
            'b-reference-etal', 'i-reference-etal'
        ], etal_sub_tag='reference-etal', etal_merge_enabled_sub_tags={
            'reference-author', 'reference-editor'
        }) == [
            'b-reference-author', 'i-reference-author',
            None,
            'b-reference-author', 'i-reference-author'
        ]

    def test_should_not_map_etal_after_other_tag(self):
        assert get_etal_mapped_tags([
            'b-reference-editor', 'i-reference-editor',
            'b-other',
            'b-reference-etal', 'i-reference-etal'
        ], etal_sub_tag='reference-etal', etal_merge_enabled_sub_tags={
            'reference-author', 'reference-editor'
        }) == [
            'b-reference-editor', 'i-reference-editor',
            'b-other',
            'b-reference-etal', 'i-reference-etal'
        ]

    def test_should_map_etal_after_author_and_editor_to_previous_tag(self):
        assert get_etal_mapped_tags([
            'b-reference-author', 'i-reference-author',
            'b-reference-etal', 'i-reference-etal',
            'b-other',
            'b-reference-editor', 'i-reference-editor',
            'b-reference-etal', 'i-reference-etal'
        ], etal_sub_tag='reference-etal', etal_merge_enabled_sub_tags={
            'reference-author', 'reference-editor'
        }) == [
            'b-reference-author', 'i-reference-author',
            'b-reference-author', 'i-reference-author',
            'b-other',
            'b-reference-editor', 'i-reference-editor',
            'b-reference-editor', 'i-reference-editor'
        ]
