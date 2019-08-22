from __future__ import absolute_import

import logging

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    TeiTagNames,
    _to_text_tokens,
    _lines_to_tei as _original_lines_to_tei,
    TeiLine,
    TeiText,
    TAG_ATTRIB_NAME,
    DEFAULT_TAG_KEY
)


LOGGER = logging.getLogger(__name__)


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'

TAG_1 = 'tag1'
TAG_2 = 'tag2'


def _tei(text_items: list = None, front_items: list = None):
    if text_items is None:
        text_items = [E.front(*front_items)]
    return E.tei(E.text(*text_items))


def _tei_lb():
    return E(TeiTagNames.LB)


def _get_all_lines(doc):
    page = doc.get_pages()[0]
    return list(doc.get_lines_of_page(page))


def _get_token_texts_for_lines(doc, lines):
    return [
        [doc.get_text(t) for t in doc.get_tokens_of_line(line)]
        for line in lines
    ]


def _tei_text(text, tag=None):
    return TeiText(text, attrib={
        TAG_ATTRIB_NAME: tag
    })


def _to_xml(node: etree.Element) -> str:
    return etree.tostring(node).decode('utf-8')


def _lines_to_tei(*args, **kwargs):
    root = _original_lines_to_tei(*args, **kwargs)
    LOGGER.debug('root: %s', _to_xml(root))
    return root


class TestToTextTokens(object):
    def test_should_not_add_space_to_single_item(self):
        assert [t.text for t in _to_text_tokens('A')] == ['A']

    def test_should_add_space_between_two_items(self):
        assert [t.text for t in _to_text_tokens('A B')] == ['A', ' ', 'B']

    def test_should_keep_preceding_space_of_item(self):
        assert [t.text for t in _to_text_tokens(' A')] == [' ', 'A']

    def test_should_keep_tailing_space_of_item(self):
        assert [t.text for t in _to_text_tokens('A ')] == ['A', ' ']


class TestGrobidTrainingStructuredDocument(object):
    def test_should_return_root_as_pages(self):
        root = _tei(front_items=[])
        doc = GrobidTrainingTeiStructuredDocument(root)
        assert list(doc.get_pages()) == [root]

    def test_should_find_one_line_with_one_token_at_front_level(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1)
            ]),
            container_node_path='text/front'
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1]
        ]
        assert doc.root.find('./text/front/note').text == TOKEN_1

    def test_should_find_one_line_with_one_token_at_text_level(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(text_items=[
                E.note(TOKEN_1)
            ]),
            container_node_path='text'
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1]
        ]
        assert doc.root.find('./text/note').text == TOKEN_1

    def test_should_find_two_lines_separated_by_lb_element(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1, _tei_lb(), TOKEN_2)
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1],
            [TOKEN_2]
        ]

    def test_should_find_empty_first_line_outside_semantic_element(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                _tei_lb(),
                E.note(TOKEN_1)
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [],
            [TOKEN_1]
        ]

    def test_should_find_empty_first_line_and_text_outside_semantic_element(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                TOKEN_1,
                _tei_lb(),
                TOKEN_2,
                E.note(TOKEN_3)
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1],
            [TOKEN_2, TOKEN_3]
        ]

    def test_should_find_text_outside_semantic_element_at_the_end(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1, E(TeiTagNames.LB)),
                TOKEN_2
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1],
            [TOKEN_2]
        ]

    def test_should_find_text_in_nested_top_level_element(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.docTitle(E.titlePart(TOKEN_1))
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1]
        ]

    def test_should_split_words_as_separate_tokens(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(' '.join([TOKEN_1, TOKEN_2]))
            ])
        )
        lines = _get_all_lines(doc)
        assert _get_token_texts_for_lines(doc, lines) == [
            [TOKEN_1, TOKEN_2]
        ]

    def test_should_be_able_to_set_tag(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1)
            ])
        )
        lines = _get_all_lines(doc)
        tokens = list(doc.get_tokens_of_line(lines[0]))
        token = tokens[0]
        doc.set_tag(token, TAG_1)
        assert doc.get_tag(token) == TAG_1

    def test_should_be_able_get_root_with_updated_single_token_tag(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1)
            ])
        )
        lines = _get_all_lines(doc)
        tokens = list(doc.get_tokens_of_line(lines[0]))
        token = tokens[0]
        doc.set_tag(token, TAG_1)
        root = doc.root
        front = root.find('./text/front')
        child_elements = list(front)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert [c.text for c in child_elements] == [TOKEN_1]

    def test_should_be_able_to_set_tag_with_attribute(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1)
            ]),
            tag_to_tei_path_mapping={TAG_1: 'div[tag="tag1"]'}
        )
        lines = _get_all_lines(doc)
        tokens = list(doc.get_tokens_of_line(lines[0]))
        token = tokens[0]
        doc.set_tag(token, TAG_1)
        assert doc.get_tag(token) == TAG_1

    def test_should_be_able_get_root_with_updated_single_token_tag_with_attribute(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                E.note(TOKEN_1)
            ]),
            tag_to_tei_path_mapping={TAG_1: 'div[tag="tag1"]'}
        )
        lines = _get_all_lines(doc)
        tokens = list(doc.get_tokens_of_line(lines[0]))
        token = tokens[0]
        doc.set_tag(token, TAG_1)
        root = doc.root
        front = root.find('./text/front')
        child_elements = list(front)
        assert [c.tag for c in child_elements] == ['div']
        assert [c.attrib for c in child_elements] == [{'tag': 'tag1'}]
        assert [c.text for c in child_elements] == [TOKEN_1]

    def test_should_preserve_space_after_lb_in_updated_root(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                TOKEN_1,
                E(TeiTagNames.LB),
                ' ' + TOKEN_2
            ])
        )
        lines = _get_all_lines(doc)

        line1_tokens = list(doc.get_tokens_of_line(lines[0]))
        doc.set_tag(line1_tokens[0], TAG_1)

        line2_tokens = list(doc.get_tokens_of_line(lines[1]))
        doc.set_tag(line2_tokens[-1], TAG_1)

        root = doc.root
        front = root.find('./text/front')
        child_elements = list(front)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert _to_xml(child_elements[0]) == (
            '<{tag1}>{token1}<{lb}/> {token2}</{tag1}>'.format(
                tag1=TAG_1, token1=TOKEN_1, token2=TOKEN_2, lb=TeiTagNames.LB
            )
        )

    def test_should_not_include_space_in_tag_if_previous_token_has_different_tag(self):
        doc = GrobidTrainingTeiStructuredDocument(
            _tei(front_items=[
                TOKEN_1,
                E(TeiTagNames.LB),
                ' ' + TOKEN_2
            ])
        )
        lines = _get_all_lines(doc)

        line1_tokens = list(doc.get_tokens_of_line(lines[0]))
        doc.set_tag(line1_tokens[0], TAG_1)

        line2_tokens = list(doc.get_tokens_of_line(lines[1]))
        doc.set_tag(line2_tokens[-1], TAG_2)

        root = doc.root
        front = root.find('./text/front')
        LOGGER.debug('xml: %s', _to_xml(front))
        assert _to_xml(front) == (
            '<front><{tag1}>{token1}<{lb}/></{tag1}>'
            ' <{tag2}>{token2}</{tag2}></front>'.format(
                tag1=TAG_1, tag2=TAG_2, token1=TOKEN_1, token2=TOKEN_2, lb=TeiTagNames.LB
            )
        )

    def test_should_preserve_existing_tag(self):
        original_tei_xml = _tei(front_items=[
            E.docTitle(E.titlePart(TOKEN_1)),
            TOKEN_2
        ])
        LOGGER.debug('original tei xml: %s', _to_xml(original_tei_xml))
        doc = GrobidTrainingTeiStructuredDocument(
            original_tei_xml,
            preserve_tags=True,
            tag_to_tei_path_mapping={}
        )
        LOGGER.debug('doc: %s', doc)

        root = doc.root
        front = root.find('./text/front')
        LOGGER.debug('xml: %s', _to_xml(front))
        assert _to_xml(front) == (
            '<front><docTitle><titlePart>{token1}</titlePart></docTitle>{token2}</front>'.format(
                token1=TOKEN_1, token2=TOKEN_2
            )
        )

    def test_should_preserve_existing_tag_with_attrib(self):
        original_tei_xml = _tei(front_items=[
            E.div(TOKEN_1, {'tag': TAG_1}),
            TOKEN_2
        ])
        LOGGER.debug('original tei xml: %s', _to_xml(original_tei_xml))
        doc = GrobidTrainingTeiStructuredDocument(
            original_tei_xml,
            preserve_tags=True,
            tag_to_tei_path_mapping={TAG_1: 'div[tag="tag1"]'}
        )
        LOGGER.debug('doc: %s', doc)

        root = doc.root
        front = root.find('./text/front')
        LOGGER.debug('xml: %s', _to_xml(front))
        assert _to_xml(front) == (
            '<front><div tag="{TAG_1}">{token1}</div>{token2}</front>'.format(
                token1=TOKEN_1, token2=TOKEN_2, TAG_1=TAG_1
            )
        )

    def test_should_not_return_preserved_tag_as_tag_and_update_preserved_tag(self):
        original_tei_xml = _tei(front_items=[
            E.note(TOKEN_1)
        ])
        LOGGER.debug('original tei xml: %s', _to_xml(original_tei_xml))
        doc = GrobidTrainingTeiStructuredDocument(
            original_tei_xml,
            preserve_tags=True,
            tag_to_tei_path_mapping={}
        )
        LOGGER.debug('doc: %s', doc)
        lines = _get_all_lines(doc)
        token1 = list(doc.get_tokens_of_line(lines[0]))[0]
        assert not doc.get_tag(token1)
        doc.set_tag(token1, TAG_1)

        root = doc.root
        front = root.find('./text/front')
        LOGGER.debug('xml: %s', _to_xml(front))
        assert _to_xml(front) == (
            '<front><{tag1}>{token1}</{tag1}></front>'.format(
                token1=TOKEN_1, tag1=TAG_1
            )
        )

    def test_should_only_preserve_tags_of_not_overlapping_lines(self):
        original_tei_xml = _tei(front_items=[
            E.note(TOKEN_1), E.note(TOKEN_2), E.lb(),
            E.note(TOKEN_3)
        ])
        LOGGER.debug('original tei xml: %s', _to_xml(original_tei_xml))
        doc = GrobidTrainingTeiStructuredDocument(
            original_tei_xml,
            preserve_tags=True,
            tag_to_tei_path_mapping={}
        )
        LOGGER.debug('doc: %s', doc)
        lines = _get_all_lines(doc)
        token1 = list(doc.get_tokens_of_line(lines[0]))[0]
        assert not doc.get_tag(token1)
        doc.set_tag(token1, TAG_1)

        root = doc.root
        front = root.find('./text/front')
        LOGGER.debug('xml: %s', _to_xml(front))
        assert _to_xml(front) == (
            '<front><{tag1}>{token1}</{tag1}>{token2}<lb/><note>{token3}</note></front>'.format(
                token1=TOKEN_1, token2=TOKEN_2, token3=TOKEN_3, tag1=TAG_1
            )
        )

    def test_should_reverse_map_tags(self):
        tag_to_tei_path_mapping = {
            TAG_1: 'docTitle/titlePart'
        }
        original_tei_xml = _tei(front_items=[
            E.docTitle(E.titlePart(TOKEN_1))
        ])
        LOGGER.debug('original tei xml: %s', _to_xml(original_tei_xml))
        doc = GrobidTrainingTeiStructuredDocument(
            original_tei_xml,
            tag_to_tei_path_mapping=tag_to_tei_path_mapping,
            preserve_tags=True
        )
        LOGGER.debug('doc: %s', doc)

        assert [
            [doc.get_tag_or_preserved_tag(t) for t in doc.get_all_tokens_of_line(line)]
            for line in _get_all_lines(doc)
        ] == [[TAG_1]]


class TestLinesToTei(object):
    def test_should_convert_single_token(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([_tei_text(TOKEN_1, TAG_1)])]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert [c.text for c in child_elements] == [TOKEN_1]

    def test_should_add_lb_element_before_token_with_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([]),
                TeiLine([_tei_text(TOKEN_1, TAG_1)])
            ]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TeiTagNames.LB, TAG_1]
        assert child_elements[1].text == TOKEN_1

    def test_should_add_lb_element_before_token_without_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([]),
                TeiLine([_tei_text(TOKEN_1, None)])
            ]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TeiTagNames.LB]
        assert child_elements[0].tail == TOKEN_1

    def test_should_add_lb_element_before_tokens_without_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([]),
                TeiLine([_tei_text(TOKEN_1, None), _tei_text(' ' + TOKEN_2, None)])
            ]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TeiTagNames.LB]
        assert child_elements[0].tail == ' '.join([TOKEN_1, TOKEN_2])

    def test_should_add_lb_within_tokens_with_same_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([_tei_text(TOKEN_1, TAG_1)]),
                TeiLine([_tei_text(TOKEN_2, TAG_1)])
            ]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert _to_xml(child_elements[0]) == (
            '<{tag1}>{token1}<{lb}/>{token2}</{tag1}>'.format(
                tag1=TAG_1, token1=TOKEN_1, token2=TOKEN_2, lb=TeiTagNames.LB
            )
        )

    def test_should_preserve_space_after_lb_within_tokens_with_same_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([_tei_text(TOKEN_1, TAG_1)]),
                TeiLine([_tei_text(' ' + TOKEN_2, TAG_1)])
            ]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert _to_xml(child_elements[0]) == (
            '<{tag1}>{token1}<{lb}/> {token2}</{tag1}>'.format(
                tag1=TAG_1, token1=TOKEN_1, token2=TOKEN_2, lb=TeiTagNames.LB
            )
        )

    def test_should_not_include_standalone_space_after_lb_in_tag_before_other_different_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [
                TeiLine([_tei_text(TOKEN_1, TAG_1)]),
                TeiLine([_tei_text(' ', None), _tei_text(TOKEN_2, TAG_2)])
            ]
        )
        assert _to_xml(tei_parent) == (
            '<front><{tag1}>{token1}<{lb}/></{tag1}> <{tag2}>{token2}</{tag2}></front>'.format(
                tag1=TAG_1, tag2=TAG_2, token1=TOKEN_1, token2=TOKEN_2, lb=TeiTagNames.LB
            )
        )

    def test_should_combine_tokens(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([
                _tei_text(TOKEN_1, TAG_1),
                _tei_text(' ' + TOKEN_2, TAG_1)
            ])]
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TAG_1]
        assert child_elements[0].text == ' '.join([TOKEN_1, TOKEN_2])

    def test_should_map_tag_to_tei_path(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([_tei_text(TOKEN_1, TAG_1)])],
            tag_to_tei_path_mapping={
                TAG_1: TAG_2
            }
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == [TAG_2]
        assert child_elements[0].text == TOKEN_1

    def test_should_map_tag_to_nested_tei_path(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([_tei_text(TOKEN_1, TAG_1)])],
            tag_to_tei_path_mapping={
                TAG_1: 'parent/child'
            }
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == ['parent']
        nested_child_elements = list(child_elements[0])
        assert [c.tag for c in nested_child_elements] == ['child']
        assert nested_child_elements[0].text == TOKEN_1

    def test_should_use_common_path_between_similar_nested_tag_paths(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([_tei_text(TOKEN_1, TAG_1), _tei_text(TOKEN_2, TAG_2)])],
            tag_to_tei_path_mapping={
                TAG_1: 'parent/child1',
                TAG_2: 'parent/child2'
            }
        )
        assert _to_xml(tei_parent) == (
            '<front><parent><child1>{token1}</child1>'
            '<child2>{token2}</child2></parent></front>'.format(
                token1=TOKEN_1, token2=TOKEN_2
            )
        )

    def test_should_apply_default_tag(self):
        tei_parent = _lines_to_tei(
            E.front(),
            [TeiLine([_tei_text(TOKEN_1, '')])],
            tag_to_tei_path_mapping={
                DEFAULT_TAG_KEY: 'other'
            }
        )
        child_elements = list(tei_parent)
        assert [c.tag for c in child_elements] == ['other']
