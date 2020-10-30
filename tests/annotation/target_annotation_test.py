from lxml.builder import E

from sciencebeam_trainer_grobid_tools.annotation.target_annotation import (
    contains_raw_text,
    get_raw_text_content,
    xml_root_to_target_annotations
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


class TestXmlRootToTargetAnnotations:
    def test_should_select_mapping_based_on_root(self):
        target_annotations = xml_root_to_target_annotations(
            E.root2(E.item1('text 1'), E.item2('text 2')),
            {
                'root1': {
                    'item': '//item1'
                },
                'root2': {
                    'item': '//item2'
                }
            }
        )
        assert [t.value for t in target_annotations] == ['text 2']

    def test_should_extract_simple_text(self):
        target_annotations = xml_root_to_target_annotations(
            E.root(E.item('text 1')),
            {'root': {
                'item': '//item'
            }}
        )
        assert [t.value for t in target_annotations] == ['text 1']

    def test_should_extract_text_including_children(self):
        target_annotations = xml_root_to_target_annotations(
            E.root(E.item('text 1 ', E.child('child text'))),
            {'root': {
                'item': '//item'
            }}
        )
        assert [t.value for t in target_annotations] == ['text 1 child text']

    def test_should_ignore_selected_children(self):
        target_annotations = xml_root_to_target_annotations(
            E.root(E.item('text 1 ', E.other('other text '), E.child('child text'))),
            {'root': {
                'item': '//item',
                'item.ignore': './/other'
            }}
        )
        assert [t.value for t in target_annotations] == ['text 1 child text']

    def test_should_ignore_selected_nested_children(self):
        target_annotations = xml_root_to_target_annotations(
            E.root(E.item(E.p('text 1 ', E.other('other text '), E.child('child text')))),
            {'root': {
                'item': '//item',
                'item.ignore': './/other'
            }}
        )
        assert [t.value for t in target_annotations] == ['text 1 child text']
