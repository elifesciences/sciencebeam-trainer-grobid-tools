from sciencebeam_trainer_grobid_tools.utils.string import (
    comma_separated_str_to_list,
    plus_minus_comma_separated_str_to_list
)


class TestCommaSeparatedStrToList:
    def test_should_parse_empty_str_as_empty_list(self):
        assert comma_separated_str_to_list('') == []

    def test_should_parse_single_item_str_as_single_item_list(self):
        assert comma_separated_str_to_list('abc') == ['abc']

    def test_should_parse_multiple_item_str(self):
        assert comma_separated_str_to_list('abc,xyz,123') == [
            'abc', 'xyz', '123'
        ]

    def test_should_strip_space_around_items(self):
        assert comma_separated_str_to_list(' abc , xyz , 123 ') == [
            'abc', 'xyz', '123'
        ]


class TestPlusMinusCommaSeparatedStrToList:
    def test_should_parse_empty_str_as_empty_list(self):
        assert plus_minus_comma_separated_str_to_list('', ['def1', 'def2']) == []

    def test_should_parse_single_item_str_as_single_item_list(self):
        assert plus_minus_comma_separated_str_to_list('abc', ['def1', 'def2']) == ['abc']

    def test_should_parse_multiple_item_str(self):
        assert plus_minus_comma_separated_str_to_list('abc,xyz,123', ['def1', 'def2']) == [
            'abc', 'xyz', '123'
        ]

    def test_should_strip_space_around_items(self):
        assert plus_minus_comma_separated_str_to_list(' abc , xyz , 123 ', ['def1', 'def2']) == [
            'abc', 'xyz', '123'
        ]

    def test_should_add_values(self):
        assert plus_minus_comma_separated_str_to_list('+abc,xyz', ['def1', 'def2']) == [
            'def1', 'def2', 'abc', 'xyz'
        ]

    def test_should_remove_values(self):
        assert plus_minus_comma_separated_str_to_list('-def2', ['def1', 'def2']) == [
            'def1'
        ]

    def test_should_add_and_remove_values(self):
        assert plus_minus_comma_separated_str_to_list(
            '+abc,xyz,-def2', ['def1', 'def2']
        ) == [
            'def1', 'abc', 'xyz'
        ]
