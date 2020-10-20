from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken
)

from sciencebeam_trainer_grobid_tools.annotation.matching_utils import (
    join_with_index_ranges,
    SequenceWrapper
)


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'


class TestJoinWithIndexRanges:
    def test_should_join_two_tokens_with_space(self):
        text, index_ranges = join_with_index_ranges([TOKEN_1, TOKEN_2], sep=' ')
        assert text == ' '.join([TOKEN_1, TOKEN_2])
        assert index_ranges == [
            (0, len(TOKEN_1)),
            (len(TOKEN_1) + 1, len(text))
        ]

    def test_should_join_two_tokens_with_no_space(self):
        text, index_ranges = join_with_index_ranges(
            [TOKEN_1, TOKEN_2],
            sep=' ',
            whitespace_list=['', ' ']
        )
        assert text == ''.join([TOKEN_1, TOKEN_2])
        assert index_ranges == [
            (0, len(TOKEN_1)),
            (len(TOKEN_1), len(text))
        ]


class TestSequenceWrapper:
    def test_should_join_text_with_space(self):
        tokens = [SimpleToken(TOKEN_1), SimpleToken(TOKEN_2)]
        doc = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        seq = SequenceWrapper(doc, tokens)
        assert str(seq) == ' '.join([TOKEN_1, TOKEN_2])

    def test_should_join_text_without_space(self):
        tokens = [SimpleToken(TOKEN_1), SimpleToken(TOKEN_2)]
        tokens[0].whitespace = ''
        doc = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        seq = SequenceWrapper(doc, tokens)
        assert str(seq) == ''.join([TOKEN_1, TOKEN_2])

    def test_should_get_tokens_between(self):
        tokens = [SimpleToken(TOKEN_1), SimpleToken(TOKEN_2)]
        doc = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        seq = SequenceWrapper(doc, tokens)
        assert list(seq.tokens_between((0, len(TOKEN_1)))) == [tokens[0]]
