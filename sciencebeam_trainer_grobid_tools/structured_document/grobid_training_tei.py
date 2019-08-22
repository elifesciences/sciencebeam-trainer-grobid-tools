from __future__ import absolute_import

import copy
import logging
import json
import re
from typing import Dict

from apache_beam.io.filesystems import FileSystems

from lxml import etree
from lxml.builder import E

from sciencebeam_utils.beam_utils.io import (
    save_file_content
)

from sciencebeam_utils.utils.xml import (
    set_or_remove_attrib
)

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    get_scoped_attrib_name,
    get_attrib_by_scope
)


TAG_ATTRIB_NAME = 'tag'
PRESERVED_TAG_ATTRIB_NAME = 'preserved_tag'


DEFAULT_TAG_KEY = ''


DEFAULT_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'note[type="other"]',
    'title': 'docTitle/titlePart',
    'abstract': 'div[type="abstract"]'
}


DEFAULT_CONTAINER_NODE_PATH = 'text/front'


class TeiTagNames(object):
    LB = 'lb'


class SvgStyleClasses(object):
    LINE = 'line'
    BLOCK = 'block'
    LINE_NO = 'line_no'


class TeiText(object):
    def __init__(self, text, tag=None, attrib=None):
        self.text = text
        self.stripped_text = text.strip()
        self.attrib = attrib if attrib is not None else {}
        if tag is not None:
            self.attrib[TAG_ATTRIB_NAME] = tag
        self.line = None

    def __repr__(self):
        return '%s(%s, tag=%s, preserved_tag=%s)' % (
            type(self).__name__, json.dumps(self.text), self.attrib.get(TAG_ATTRIB_NAME),
            self.attrib.get(PRESERVED_TAG_ATTRIB_NAME)
        )


class TeiSpace(TeiText):
    pass


class TeiLine(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.non_space_tokens = [t for t in tokens if not isinstance(t, TeiSpace)]
        for token in tokens:
            token.line = self

    def __repr__(self):
        return 'TeiLine(%s)' % self.tokens


class LineBuffer(object):
    def __init__(self):
        self.line_tokens = []

    def append(self, token):
        self.line_tokens.append(token)

    def extend(self, tokens):
        self.line_tokens.extend(tokens)

    def flush(self):
        line = TeiLine(self.line_tokens)
        self.line_tokens = []
        return line

    def __len__(self):
        return len(self.line_tokens)


class TagExpression(object):
    def __init__(self, tag, attrib):
        self.tag = tag
        self.attrib = attrib

    def create_node(self):
        return E(self.tag, self.attrib)


def get_logger():
    return logging.getLogger(__name__)


def _to_text_tokens(text, tag=None):
    if not text:
        return []
    return [
        TeiText(s, tag=tag)
        if s.strip()
        else TeiSpace(s, tag=tag)
        for s in re.split(r'(\W)', text)
        if s
    ]


def _parse_tag_expression(tag_expression):
    match = re.match(r'^([^\[]+)(\[([^=]+)="(.+)"\])?$', tag_expression)
    if not match:
        raise ValueError('invalid tag expression: %s' % tag_expression)
    get_logger().debug('match: %s', match.groups())
    tag_name = match.group(1)
    if match.group(2):
        attrib = {match.group(3): match.group(4)}
    else:
        attrib = {}
    return TagExpression(tag=tag_name, attrib=attrib)


def _node_to_tag_expression(node):
    if not node.attrib:
        return node.tag
    if len(node.attrib) > 1:
        raise ValueError('only supporting up to one attribute')
    key, value = list(node.attrib.items())[0]
    return '{tag}[{key}="{value}"]'.format(tag=node.tag, key=key, value=value)


def _iter_extract_lines_from_element(parent_element, line_buffer, current_path):
    current_tag = '/'.join(current_path) if current_path else None
    line_buffer.extend(_to_text_tokens(parent_element.text, current_tag))

    for child_element in parent_element:
        if child_element.tag == TeiTagNames.LB:
            yield line_buffer.flush()

        child_path = current_path + [_node_to_tag_expression(child_element)]
        for line in _iter_extract_lines_from_element(child_element, line_buffer, child_path):
            yield line

    parent_tag = '/'.join(current_path[:-1]) if len(current_path) >= 2 else None
    line_buffer.extend(_to_text_tokens(parent_element.tail, parent_tag))


def _iter_extract_lines_from_container_elements(container_elements):
    line_buffer = LineBuffer()
    current_path = []

    for container_element in container_elements:
        lines = _iter_extract_lines_from_element(container_element, line_buffer, current_path)
        for line in lines:
            yield line
        if line_buffer:
            yield line_buffer.flush()


def _get_tag_attrib_name(scope, level):
    return get_scoped_attrib_name(TAG_ATTRIB_NAME, scope=scope, level=level)


def _get_last_child_or_none(element):
    try:
        return element[-1]
    except IndexError:
        return None


def _append_text(element, text):
    if not text:
        return
    last_child = _get_last_child_or_none(element)
    if last_child is not None and last_child.tail:
        last_child.tail = last_child.tail + '' + text
    elif last_child is not None:
        last_child.tail = text
    elif element.text:
        element.text = element.text + '' + text
    else:
        element.text = text


def _get_common_path(path1, path2):
    if path1 == path2:
        return path1
    for path_len in range(1, 1 + min(len(path1), len(path2))):
        if path1[:path_len] == path2[:path_len]:
            return path1[:path_len]
    return []


def _get_element_at_path(current_element, current_path, required_path, token):
    if required_path != current_path:
        common_path = _get_common_path(current_path, required_path)
        get_logger().debug(
            'required element path: %s -> %s (%s, [%s])',
            current_path, required_path, common_path, token.text
        )
        for _ in range(len(current_path) - len(common_path)):
            current_element = current_element.getparent()
        current_path = common_path
        for path_fragment in required_path[len(common_path):]:
            parsed_path_fragment = _parse_tag_expression(path_fragment)
            child = parsed_path_fragment.create_node()
            current_element.append(child)
            current_element = child
            current_path.append(path_fragment)
    return current_element, current_path


def _lines_to_tei(
        parent: etree.Element,
        lines: list,
        tag_to_tei_path_mapping: Dict[str, str] = None):
    if tag_to_tei_path_mapping is None:
        tag_to_tei_path_mapping = {}
    current_element = parent
    current_path = []
    pending_space_tokens = []
    for i, line in enumerate(lines):
        if i:
            current_element.append(E(TeiTagNames.LB))
        for token in line.tokens:
            tag = token.attrib.get(TAG_ATTRIB_NAME)
            if not tag:
                tag = token.attrib.get(PRESERVED_TAG_ATTRIB_NAME)
            if tag:
                required_path = tag_to_tei_path_mapping.get(tag, tag).split('/')
            else:
                if not token.stripped_text:
                    pending_space_tokens.append(token)
                    continue
                required_path = []
                default_path_str = tag_to_tei_path_mapping.get(DEFAULT_TAG_KEY)
                if default_path_str:
                    required_path = default_path_str.split('/')

            for pending_space_token in pending_space_tokens:
                current_element, current_path = _get_element_at_path(
                    current_element, current_path,
                    _get_common_path(current_path, required_path),
                    pending_space_token
                )
                _append_text(current_element, pending_space_token.text)
                pending_space_tokens = []

            current_element, current_path = _get_element_at_path(
                current_element, current_path, required_path, token
            )

            _append_text(current_element, token.text)

    for pending_space_token in pending_space_tokens:
        _append_text(current_element, pending_space_token.text)
        pending_space_tokens = []

    return parent


def _updated_tei_with_lines(
        original_root: etree.Element,
        lines: list,
        tag_to_tei_path_mapping: Dict[str, str],
        container_node_path: str = 'text/front'):
    updated_root = copy.deepcopy(original_root)
    container_node = updated_root.find(container_node_path)
    get_logger().debug('container_node: %s', container_node)
    container_node.clear()
    _lines_to_tei(container_node, lines, tag_to_tei_path_mapping)
    return updated_root


class GrobidTrainingTeiStructuredDocument(AbstractStructuredDocument):
    def __init__(
            self,
            root: etree.Element,
            tag_to_tei_path_mapping: Dict[str, str] = None,
            preserve_tags: bool = True,
            container_node_path: str = DEFAULT_CONTAINER_NODE_PATH):
        self._root = root
        self._container_node_path = container_node_path
        self._lines = list(_iter_extract_lines_from_container_elements(
            root.findall('./%s' % container_node_path)
        ))
        self._tag_to_tei_path_mapping = (
            tag_to_tei_path_mapping if tag_to_tei_path_mapping is not None
            else DEFAULT_TAG_TO_TEI_PATH_MAPPING
        )
        rev_tag_to_tei_path_mapping = {v: k for k, v in self._tag_to_tei_path_mapping.items()}
        if preserve_tags:
            for line in self._lines:
                for token in line.tokens:
                    existing_tag = self.get_tag(token)
                    mapped_tag = rev_tag_to_tei_path_mapping.get(existing_tag, existing_tag)
                    self._set_preserved_tag(token, mapped_tag)
        for line in self._lines:
            for token in line.tokens:
                self.set_tag_only(token, None)

    @property
    def root(self):
        return _updated_tei_with_lines(
            self._root,
            self._lines,
            self._tag_to_tei_path_mapping,
            container_node_path=self._container_node_path
        )

    def get_pages(self):
        return [self._root]

    def get_lines_of_page(self, page):
        if page != self._root:
            return []
        return self._lines

    def get_tokens_of_line(self, line):
        return line.non_space_tokens

    def get_all_tokens_of_line(self, line):
        return line.tokens

    def get_x(self, parent):
        raise NotImplementedError()

    def get_text(self, parent):
        return parent.stripped_text

    def get_tag(self, parent, scope=None, level=None):
        return parent.attrib.get(_get_tag_attrib_name(scope, level))

    def get_tag_or_preserved_tag(self, parent, scope=None, level=None):
        tag = self.get_tag(parent, scope=scope, level=level)
        if not tag:
            tag = parent.attrib.get(PRESERVED_TAG_ATTRIB_NAME)
        return tag

    def set_tag_only(self, parent, tag, scope=None, level=None):
        set_or_remove_attrib(parent.attrib, _get_tag_attrib_name(scope, level), tag)

    def set_tag(self, parent, tag, scope=None, level=None):
        self.set_tag_only(parent, tag, scope=scope, level=level)
        if not isinstance(parent, TeiSpace):
            self._clear_same_preserved_tag_on_same_line(parent)

    def _clear_same_preserved_tag_on_same_line(self, token):
        preserved_tag = token.attrib.get(PRESERVED_TAG_ATTRIB_NAME)
        if not preserved_tag:
            return
        line_tokens = token.line.tokens
        get_logger().debug('clearing tokens on same line: %s (%s)', preserved_tag, line_tokens)
        for line_token in line_tokens:
            if line_token.attrib.get(PRESERVED_TAG_ATTRIB_NAME) == preserved_tag:
                self._set_preserved_tag(line_token, None)

    def _set_preserved_tag(self, parent, tag):
        set_or_remove_attrib(parent.attrib, PRESERVED_TAG_ATTRIB_NAME, tag)

    def get_tag_by_scope(self, parent):
        return get_attrib_by_scope(parent.attrib, TAG_ATTRIB_NAME)

    def get_bounding_box(self, parent):
        raise NotImplementedError()

    def set_bounding_box(self, parent, bounding_box):
        raise NotImplementedError()

    def __repr__(self):
        return '%s(lines=%s)' % (type(self).__name__, self._lines)


def load_xml_root(filename, **kwargs):
    with FileSystems.open(filename) as f:
        return etree.parse(f, **kwargs).getroot()


def load_grobid_training_tei_structured_document(
        filename: str,
        container_node_path: str = DEFAULT_CONTAINER_NODE_PATH):
    parser = etree.XMLParser(recover=True)
    return GrobidTrainingTeiStructuredDocument(
        load_xml_root(filename, parser=parser),
        container_node_path=container_node_path
    )


def save_grobid_training_tei_structured_document(
        filename, grobid_training_tei_structured_document):
    save_file_content(
        filename, etree.tostring(grobid_training_tei_structured_document.root)
    )
