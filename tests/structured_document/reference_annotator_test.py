from sciencebeam_trainer_grobid_tools.structured_document.reference_annotator import (
    get_prefix_extended_token_tags
)


class TestPrefixExtendedTokenTags:
    def test_should_extend_doi_prefix(self):
        assert get_prefix_extended_token_tags(
            [None, None, 'b-reference-doi'],
            ['DOI', ':', '12345'],
            enabled_tags={'reference-doi'}
        ) == ['b-reference-doi', 'i-reference-doi', 'i-reference-doi']

    def test_not_should_extend_to_other_prefix_text(self):
        assert get_prefix_extended_token_tags(
            [None, None, None, None, None, None, 'b-reference-doi'],
            ['some', 'other', 'text', ',', 'DOI', ':', '12345'],
            enabled_tags={'reference-doi'}
        ) == [
            None, None, None, None,
            'b-reference-doi', 'i-reference-doi', 'i-reference-doi'
        ]

    def test_should_not_extend_other_tag(self):
        assert get_prefix_extended_token_tags(
            [None, None, 'b-other'],
            ['DOI', ':', '12345'],
            enabled_tags={'reference-doi'}
        ) == [None, None, 'b-other']
