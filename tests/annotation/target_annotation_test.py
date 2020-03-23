from lxml.builder import E

from sciencebeam_trainer_grobid_tools.annotation.target_annotation import (
    contains_raw_text,
    get_raw_text_content
)


class TestContainsRawTextContent:
    def test_should_return_true_if_element_contains_text(self):
        assert contains_raw_text(E.node('raw text 1'))

    def test_should_return_false_if_element_contains_child_element_with_text(self):
        assert not contains_raw_text(E.node(E.child('raw text 1')))

    def test_should_return_true_if_child_element_is_followed_by_text(self):
        assert contains_raw_text(E.node(E.child('child'), 'tail text'))

    def test_should_return_true_if_child_element_contains_child_element_followed_by_text(self):
        assert contains_raw_text(E.node(E.child(E.innerChild('child'), 'tail text')))


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
