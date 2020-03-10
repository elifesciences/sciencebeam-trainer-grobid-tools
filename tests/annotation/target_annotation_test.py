from lxml.builder import E

from sciencebeam_trainer_grobid_tools.annotation.target_annotation import (
    get_raw_text_content
)


class TestGetRawTextContent:
    def test_should_return_raw_text(self):
        assert get_raw_text_content(
            E.node('raw text 1')
        ) == 'raw text 1'

    def test_should_add_space_after_element_if_followed_by_word(self):
        assert get_raw_text_content(
            E.node(E.label('1'), 'raw text 1')
        ) == '1 raw text 1'

    def test_should_not_add_space_after_element_if_followed_by_comma(self):
        assert get_raw_text_content(
            E.node(E.label('1'), ', raw text 1')
        ) == '1, raw text 1'
