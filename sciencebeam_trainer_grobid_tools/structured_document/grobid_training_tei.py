from __future__ import absolute_import

import logging
import json
import re

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


DEFAULT_TAG_TO_TEI_PATH_MAPPING = {
    'title': 'docTitle/titlePart',
    'abstract': 'div[type="abstract"]'
}


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

    def __repr__(self):
        return '%s(%s, tag=%s)' % (
            type(self).__name__, json.dumps(self.text), self.attrib.get(TAG_ATTRIB_NAME)
        )


class TeiSpace(TeiText):
    pass


class TeiLine(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.non_space_tokens = [t for t in tokens if not isinstance(t, TeiSpace)]

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


def _lines_to_tei(tag_name, lines, tag_to_tei_path_mapping=None):
    if tag_to_tei_path_mapping is None:
        tag_to_tei_path_mapping = {}
    parent = E(tag_name)
    current_element = parent
    current_path = []
    pending_space_tokens = []
    for i, line in enumerate(lines):
        if i:
            current_element.append(E(TeiTagNames.LB))
        for token in line.tokens:
            tag = token.attrib.get(TAG_ATTRIB_NAME)
            if tag:
                required_path = tag_to_tei_path_mapping.get(tag, tag).split('/')
            else:
                if not token.stripped_text:
                    pending_space_tokens.append(token)
                    continue
                required_path = []

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


def _updated_tei_with_lines(original_root, lines, tag_to_tei_path_mapping):
    return E(
        original_root.tag,
        *[
            E.text(_lines_to_tei('front', lines, tag_to_tei_path_mapping))
            if element.tag == 'text'
            else element
            for element in original_root
        ]
    )


class GrobidTrainingTeiStructuredDocument(AbstractStructuredDocument):
    def __init__(self, root, tag_to_tei_path_mapping=None, preserve_tags=True):
        self._root = root
        self._lines = list(_iter_extract_lines_from_container_elements(
            root.findall('./text/front')
        ))
        self._tag_to_tei_path_mapping = (
            tag_to_tei_path_mapping if tag_to_tei_path_mapping is not None
            else DEFAULT_TAG_TO_TEI_PATH_MAPPING
        )
        rev_tag_to_tei_path_mapping = {v: k for k, v in self._tag_to_tei_path_mapping.items()}
        if not preserve_tags:
            for line in self._lines:
                for token in line.tokens:
                    self.set_tag(token, None)
        else:
            for line in self._lines:
                for token in line.tokens:
                    existing_tag = self.get_tag(token)
                    mapped_tag = rev_tag_to_tei_path_mapping.get(existing_tag, existing_tag)
                    if mapped_tag != existing_tag:
                        self.set_tag(token, mapped_tag)

    @property
    def root(self):
        return _updated_tei_with_lines(
            self._root,
            self._lines,
            self._tag_to_tei_path_mapping
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

    def set_tag(self, parent, tag, scope=None, level=None):
        set_or_remove_attrib(parent.attrib, _get_tag_attrib_name(scope, level), tag)

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


def load_grobid_training_tei_structured_document(filename):
    parser = etree.XMLParser(recover=True)
    return GrobidTrainingTeiStructuredDocument(
        load_xml_root(filename, parser=parser)
    )


def save_grobid_training_tei_structured_document(
        filename, grobid_training_tei_structured_document):
    save_file_content(
        filename, etree.tostring(grobid_training_tei_structured_document.root)
    )
