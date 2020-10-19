from __future__ import absolute_import

import copy
import logging
import re
from itertools import zip_longest
from typing import Dict, Iterable, List, Set

import regex
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
    get_attrib_by_scope,
    B_TAG_PREFIX,
    I_TAG_PREFIX,
    add_tag_prefix,
    split_tag_prefix,
    strip_tag_prefix
)

from sciencebeam_trainer_grobid_tools.utils.string import is_blank


LOGGER = logging.getLogger(__name__)


TAG_ATTRIB_NAME = 'tag'
SUB_LEVEL = 2
SUB_TAG_ATTRIB_NAME = get_scoped_attrib_name(TAG_ATTRIB_NAME, level=SUB_LEVEL)
PRESERVED_TAG_ATTRIB_NAME = 'preserved_tag'
PRESERVED_SUB_TAG_ATTRIB_NAME = get_scoped_attrib_name(
    PRESERVED_TAG_ATTRIB_NAME,
    level=SUB_LEVEL
)


DEFAULT_TAG_KEY = ''


DEFAULT_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'note[type="other"]',
    'title': 'docTitle/titlePart',
    'abstract': 'div[type="abstract"]'
}


class ContainerNodePaths:
    HEADER_CONTAINER_NODE_PATH = 'text/front'
    SEGMENTATION_CONTAINER_NODE_PATH = 'text'


DEFAULT_CONTAINER_NODE_PATH = ContainerNodePaths.HEADER_CONTAINER_NODE_PATH


class TeiTagNames(object):
    LB = 'lb'


class SvgStyleClasses(object):
    LINE = 'line'
    BLOCK = 'block'
    LINE_NO = 'line_no'


class TeiText(object):
    def __init__(
            self,
            text: str,
            tag: str = None,
            sub_tag: str = None,
            whitespace: str = None,
            attrib: Dict[str, str] = None):
        self.text = text
        self.stripped_text = text.strip()
        self.attrib = attrib if attrib is not None else {}
        if tag is not None:
            self.attrib[TAG_ATTRIB_NAME] = tag
        if sub_tag is not None:
            self.attrib[SUB_TAG_ATTRIB_NAME] = sub_tag
        self.line = None
        self.whitespace = whitespace

    @property
    def tag(self):
        return self.attrib.get(TAG_ATTRIB_NAME)

    def __repr__(self):
        return (
            '%s(%s, tag=%s, sub_tag=%s, preserved_tag=%s, preserved_sub_tag=%s, whitespace=%s)'
        ) % (
            type(self).__name__,
            repr(self.text),
            self.attrib.get(TAG_ATTRIB_NAME),
            self.attrib.get(SUB_TAG_ATTRIB_NAME),
            self.attrib.get(PRESERVED_TAG_ATTRIB_NAME),
            self.attrib.get(PRESERVED_SUB_TAG_ATTRIB_NAME),
            repr(self.whitespace)
        )


class TeiSpace(TeiText):
    pass


class TeiLine(object):
    def __init__(self, tokens: List[TeiText]):
        self.tokens = tokens
        self.non_space_tokens = [t for t in tokens if not isinstance(t, TeiSpace)]
        for token in tokens:
            token.line = self

    def __repr__(self):
        return 'TeiLine(%s)' % self.tokens


class TokenWriter:
    def __init__(self):
        self.tokens = []
        self.next_tag = None
        self.next_sub_tag = None

    def reset_next_tag(self):
        self.next_tag = None

    def reset_next_sub_tag(self):
        self.next_sub_tag = None

    def set_next_tag(self, tag: str, begin_tag: bool = True):
        self.next_tag = add_tag_prefix(
            tag,
            prefix=B_TAG_PREFIX if begin_tag else I_TAG_PREFIX
        )

    def set_next_sub_tag(self, tag: str, begin_tag: bool = True):
        self.next_sub_tag = add_tag_prefix(
            tag,
            prefix=B_TAG_PREFIX if begin_tag else I_TAG_PREFIX
        )

    def reset(self):
        self.tokens = []

    def append_tokenized_text_token(self, tokenized_text_token: str):
        if self.tokens:
            if not tokenized_text_token.strip():
                self.tokens[-1].whitespace = tokenized_text_token
            else:
                self.tokens[-1].whitespace = ''
        self.append(_to_text_token(
            tokenized_text_token,
            tag=self.next_tag,
            sub_tag=self.next_sub_tag
        ))
        self.next_tag = add_tag_prefix(
            strip_tag_prefix(self.next_tag),
            I_TAG_PREFIX
        )
        self.next_sub_tag = add_tag_prefix(
            strip_tag_prefix(self.next_sub_tag),
            I_TAG_PREFIX
        )

    def append_tokenized_text(self, tokenized_text: List[str]):
        for tokenized_text_token in tokenized_text:
            self.append_tokenized_text_token(tokenized_text_token)

    def append_text(self, text: str):
        if not text:
            return
        self.append_tokenized_text(_tokenize_text(text))

    def append(self, token: TeiText):
        self.tokens.append(token)

    def extend(self, tokens: List[TeiText]):
        self.tokens.extend(tokens)

    def __len__(self):
        return len(self.tokens)


class LineBuffer(TokenWriter):
    def flush(self):
        line = TeiLine(self.tokens)
        self.reset()
        return line


class TagExpression(object):
    def __init__(self, tag: str, attrib: Dict[str, str]):
        self.tag = tag
        self.attrib = attrib

    def create_node(self):
        try:
            return E(self.tag, self.attrib)
        except ValueError as e:
            raise ValueError(
                'failed to create node with tag=%r, attrib=%r due to %s' % (
                    self.tag, self.attrib, e
                )
            ) from e


def get_logger():
    return logging.getLogger(__name__)


def _iter_split_lower_to_upper_case(text: str) -> List[str]:
    start = 0
    for index, c in enumerate(text):
        if index > 0 and c.isupper() and text[index - 1].islower():
            yield text[start:index]
            start = index
    if start < len(text):
        yield text[start:]


def _tokenize_text(text: str) -> List[str]:
    return [
        s
        for s1 in regex.split(r'(\W)', text)
        for s in _iter_split_lower_to_upper_case(s1)
        if s
    ]


def _to_text_token(text: str, *args, **kwargs) -> TeiText:
    if not text.strip():
        return TeiSpace(text, *args, **kwargs)
    return TeiText(text, *args, **kwargs)


def _parse_tag_expression(tag_expression):
    match = re.match(r'^([^\[]+)(\[@?([^=]+)="(.+)"\])?$', tag_expression)
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


def has_direct_text(element: etree.Element) -> bool:
    if not is_blank(element.text):
        return True
    for child in element:
        if not is_blank(child.tail):
            return True
    return False


def _iter_extract_lines_from_element(
        parent_element: etree.Element,
        line_buffer: LineBuffer,
        current_path: List[str],
        root_paths: Set[str],
        parent_tagged_path: List[str] = None,
        begin_tag: bool = True) -> Iterable[TeiLine]:
    previous_tag = line_buffer.next_tag
    current_tag = '/'.join(current_path) if current_path else None
    if has_direct_text(parent_element) or current_tag in root_paths:
        if not previous_tag:
            line_buffer.set_next_tag(current_tag)
        else:
            line_buffer.set_next_sub_tag(current_tag)
    line_buffer.append_text(parent_element.text)

    LOGGER.debug('parent_tagged_path: %s', parent_tagged_path)
    LOGGER.debug('current_path: %s', current_path)

    for child_element in parent_element:
        if child_element.tag == TeiTagNames.LB:
            yield line_buffer.flush()

        child_path = current_path + [_node_to_tag_expression(child_element)]
        yield from _iter_extract_lines_from_element(
            child_element,
            line_buffer,
            child_path,
            root_paths=root_paths,
            parent_tagged_path=parent_tagged_path,
            begin_tag=begin_tag
        )

    if not previous_tag:
        line_buffer.reset_next_tag()
    else:
        line_buffer.reset_next_sub_tag()
    line_buffer.append_text(parent_element.tail)


def _iter_extract_lines_from_container_elements(
        container_elements: Iterable[etree.Element],
        **kwargs):
    line_buffer = LineBuffer()
    current_path = []

    for container_element in container_elements:
        lines = _iter_extract_lines_from_element(
            container_element, line_buffer, current_path, **kwargs
        )
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


def _get_common_path(path1: List[str], path2: List[str]) -> List[str]:
    if path1 == path2:
        return path1
    common_path = []
    for path1_element, path2_element in zip_longest(path1, path2):
        if path1_element != path2_element:
            break
        common_path.append(path1_element)
    return common_path


def _path_starts_with(path1: List[str], path2: List[str]) -> bool:
    return _get_common_path(path1, path2 or []) == path1


def _get_element_at_path(current_element, current_path, required_path, token):
    if required_path != current_path:
        common_path = _get_common_path(current_path, required_path)
        get_logger().debug(
            'required element path: %s -> %s (common path: %s, token text: %r)',
            current_path, required_path, common_path, token.text
        )
        for _ in range(len(current_path) - len(common_path)):
            current_element = current_element.getparent()
        current_path = common_path
        for path_fragment in required_path[len(common_path):]:
            try:
                parsed_path_fragment = _parse_tag_expression(path_fragment)
                child = parsed_path_fragment.create_node()
            except ValueError as e:
                raise ValueError('failed to create node for %r due to %s' % (
                    path_fragment, e
                )) from e
            current_element.append(child)
            current_element = child
            current_path.append(path_fragment)
    return current_element, current_path


def _split_path(path_str: str) -> List[str]:
    return regex.split(r'(?<!\{[^}]*)/', path_str)


def _get_tag_required_path(
        tag: str,
        tag_to_tei_path_mapping: Dict[str, str] = None) -> List[str]:
    if tag:
        required_path = _split_path(tag_to_tei_path_mapping.get(tag, tag))
    else:
        required_path = []
        default_path_str = tag_to_tei_path_mapping.get(DEFAULT_TAG_KEY)
        if default_path_str:
            required_path = _split_path(default_path_str)
    return required_path


class XmlTreeWriter:
    def __init__(self, parent: etree.Element):
        self.current_element = parent
        self.current_path = []

    def append(self, element: etree.Element):
        self.current_element.append(element)

    def append_text(self, text: str):
        _append_text(self.current_element, text)

    def require_path(self, required_path: List[str], token: TeiText):
        self.current_element, self.current_path = _get_element_at_path(
            self.current_element, self.current_path,
            required_path,
            token
        )

    def require_path_or_below(self, required_path: List[str], token: TeiText):
        self.require_path(
            _get_common_path(self.current_path, required_path),
            token=token
        )


def _lines_to_tei(
        parent: etree.Element,
        lines: List[TeiLine],
        tag_to_tei_path_mapping: Dict[str, str] = None):
    if tag_to_tei_path_mapping is None:
        tag_to_tei_path_mapping = {}
    writer = XmlTreeWriter(parent)
    pending_space_tokens = []
    for line_index, line in enumerate(lines):
        if line_index:
            writer.append(E(TeiTagNames.LB))
        for token in line.tokens:
            if not token.stripped_text:
                pending_space_tokens.append(token)
                continue
            main_full_tag = token.attrib.get(TAG_ATTRIB_NAME)
            if not main_full_tag:
                main_full_tag = token.attrib.get(PRESERVED_TAG_ATTRIB_NAME)
            sub_full_tag = token.attrib.get(SUB_TAG_ATTRIB_NAME)
            if not sub_full_tag:
                sub_full_tag = token.attrib.get(PRESERVED_SUB_TAG_ATTRIB_NAME)
            main_prefix, main_tag = split_tag_prefix(main_full_tag)
            sub_prefix, sub_tag = split_tag_prefix(sub_full_tag)
            main_required_path = _get_tag_required_path(main_tag, tag_to_tei_path_mapping)
            sub_required_path = (
                _get_tag_required_path(sub_tag, tag_to_tei_path_mapping)
                if sub_full_tag
                else None
            )
            if sub_full_tag and not _path_starts_with(main_required_path, sub_required_path):
                LOGGER.debug(
                    'ignoring sub tag outside main path: %s (%s)',
                    sub_tag, sub_required_path
                )
                sub_tag = None
                sub_full_tag = None
                sub_required_path = []
            LOGGER.debug(
                'output token: %s (main_required_path: %s, sub_required_path: %s)',
                token, main_required_path, sub_required_path
            )

            if main_prefix == B_TAG_PREFIX:
                LOGGER.debug('found begin prefix, resetting path: %s', main_full_tag)
                writer.require_path([], token=token)
            elif sub_prefix == B_TAG_PREFIX:
                LOGGER.debug('found begin sub prefix, resetting path to parent: %s', sub_full_tag)
                writer.require_path_or_below(main_required_path, token=token)

            required_path = (
                sub_required_path if sub_full_tag
                else main_required_path
            )

            if pending_space_tokens:
                for pending_space_token in pending_space_tokens:
                    writer.require_path_or_below(required_path, token=pending_space_token)
                    writer.append_text(pending_space_token.text)
                    pending_space_tokens = []

            writer.require_path(required_path, token=token)
            writer.append_text(token.text)

    for pending_space_token in pending_space_tokens:
        writer.require_path_or_below([], token=pending_space_token)
        writer.append_text(pending_space_token.text)
        pending_space_tokens = []

    return parent


def _updated_tei_with_lines(
        original_root: etree.Element,
        lines: list,
        tag_to_tei_path_mapping: Dict[str, str],
        container_node_path: str = 'text/front',
        namespaces: Dict[str, str] = None):
    updated_root = copy.deepcopy(original_root)
    container_node = updated_root.find(container_node_path, namespaces=namespaces)
    get_logger().debug('container_node: %s', container_node)
    if container_node is None:
        raise RuntimeError('container node path not found: %s (namespaces=%s) (has %s)' % (
            container_node_path, namespaces, list(updated_root)
        ))
    container_node.clear()
    _lines_to_tei(container_node, lines, tag_to_tei_path_mapping)
    return updated_root


class GrobidTrainingTeiStructuredDocument(AbstractStructuredDocument):
    def __init__(
            self,
            root: etree.Element,
            tag_to_tei_path_mapping: Dict[str, str] = None,
            preserve_tags: bool = True,
            container_node_path: str = DEFAULT_CONTAINER_NODE_PATH,
            namespaces: Dict[str, str] = None):
        self._root = root
        self._container_node_path = container_node_path
        self._tag_to_tei_path_mapping = (
            tag_to_tei_path_mapping if tag_to_tei_path_mapping is not None
            else DEFAULT_TAG_TO_TEI_PATH_MAPPING
        )
        self._namespaces = namespaces
        self._lines = list(_iter_extract_lines_from_container_elements(
            root.findall('./%s' % container_node_path, namespaces=namespaces),
            root_paths=self._tag_to_tei_path_mapping.values()
        ))
        if preserve_tags:
            self._preserve_current_tags()
        else:
            LOGGER.debug('not preserving tei tags')
        self._reset_current_tags()

    def _preserve_current_tags(self):
        rev_tag_to_tei_path_mapping = {v: k for k, v in self._tag_to_tei_path_mapping.items()}
        LOGGER.debug(
            'preserving tei tags using rev_tag_to_tei_path_mapping: %s',
            rev_tag_to_tei_path_mapping
        )
        for line in self._lines:
            for token in line.tokens:
                for level in (None, SUB_LEVEL):
                    full_existing_tag = self.get_tag(token, level=level)
                    prefix, existing_tag = split_tag_prefix(full_existing_tag)
                    mapped_tag = add_tag_prefix(
                        rev_tag_to_tei_path_mapping.get(existing_tag, existing_tag),
                        prefix=prefix
                    )
                    self._set_preserved_tag(token, mapped_tag, level=level)

    def _reset_current_tags(self):
        for line in self._lines:
            for token in line.tokens:
                self.set_tag_only(token, None)
                self.set_sub_tag_only(token, None)

    @property
    def root(self):
        return _updated_tei_with_lines(
            self._root,
            self._lines,
            self._tag_to_tei_path_mapping,
            container_node_path=self._container_node_path,
            namespaces=self._namespaces
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

    def get_whitespace(self, parent):
        return parent.whitespace

    def get_tag(self, parent, scope=None, level=None):
        return parent.attrib.get(_get_tag_attrib_name(scope, level))

    def get_preserved_tag(self, parent, **kwargs):
        return parent.attrib.get(get_scoped_attrib_name(PRESERVED_TAG_ATTRIB_NAME, **kwargs))

    def get_tag_or_preserved_tag(self, parent, **kwargs):
        return self.get_tag(parent, **kwargs) or self.get_preserved_tag(parent, **kwargs)

    def get_tag_or_preserved_tag_value(self, *args, **kwargs):
        return strip_tag_prefix(self.get_tag_or_preserved_tag(*args, **kwargs))

    def set_tag_only(
            self, parent: TeiText, tag: str,
            scope: str = None, level: str = None):
        set_or_remove_attrib(parent.attrib, _get_tag_attrib_name(scope, level), tag)

    def set_sub_tag_only(
            self, parent: TeiText, tag: str):
        set_or_remove_attrib(parent.attrib, SUB_TAG_ATTRIB_NAME, tag)

    def clear_preserved_sub_tag(
            self, parent: TeiText):
        set_or_remove_attrib(parent.attrib, PRESERVED_SUB_TAG_ATTRIB_NAME, None)

    def clear_preserved_tag_only(
            self, parent: TeiText, **kwargs):
        set_or_remove_attrib(
            parent.attrib,
            get_scoped_attrib_name(PRESERVED_TAG_ATTRIB_NAME, **kwargs),
            None
        )

    def set_tag(self, parent, tag, scope=None, level=None):
        _previous_tag = self.get_tag_or_preserved_tag(parent, level=level)
        self.set_tag_only(parent, tag, scope=scope, level=level)
        if isinstance(parent, TeiSpace):
            return
        if strip_tag_prefix(tag) != strip_tag_prefix(_previous_tag):
            self._clear_same_preserved_tag_on_same_line(parent, level=level)
            if level is None:
                self._clear_same_preserved_tag_on_same_line(parent, level=SUB_LEVEL)

    def _clear_same_preserved_tag_on_same_line(self, token, level: int = None):
        preserved_tag_attrib_name = get_scoped_attrib_name(PRESERVED_TAG_ATTRIB_NAME, level=level)
        preserved_tag = strip_tag_prefix(token.attrib.get(preserved_tag_attrib_name))
        if not preserved_tag:
            return
        line_tokens = token.line.tokens
        get_logger().debug('clearing tokens on same line: %s (%s)', preserved_tag, line_tokens)
        for line_token in line_tokens:
            if strip_tag_prefix(line_token.attrib.get(preserved_tag_attrib_name)) == preserved_tag:
                self._set_preserved_tag(line_token, None, level=level)

    def _set_preserved_tag(self, parent, tag, level: int = None):
        set_or_remove_attrib(
            parent.attrib,
            get_scoped_attrib_name(PRESERVED_TAG_ATTRIB_NAME, level=level),
            tag
        )

    def get_tag_by_scope(self, parent):
        return get_attrib_by_scope(parent.attrib, TAG_ATTRIB_NAME)

    def get_bounding_box(self, parent):
        raise NotImplementedError()

    def set_bounding_box(self, parent, bounding_box):
        raise NotImplementedError()

    def remove_all_untagged(self):
        updated_lines = []
        current_result_line_tokens = []
        output_enabled = True
        for line in self._lines:
            for token in line.tokens:
                if isinstance(token, TeiSpace):
                    if output_enabled:
                        current_result_line_tokens.append(token)
                    continue
                output_enabled = (token.tag or '') != ''
                if not output_enabled:
                    continue
                current_result_line_tokens.append(token)
            if current_result_line_tokens:
                updated_lines.append(TeiLine(current_result_line_tokens))
        self._lines = updated_lines

    def __repr__(self):
        return '%s(lines=%s)' % (type(self).__name__, self._lines)


def load_xml_root(filename, **kwargs):
    with FileSystems.open(filename) as f:
        return etree.parse(f, **kwargs).getroot()


def load_grobid_training_tei_structured_document(filename: str, **kwargs):
    parser = etree.XMLParser(recover=True)
    return GrobidTrainingTeiStructuredDocument(load_xml_root(filename, parser=parser), **kwargs)


def save_grobid_training_tei_structured_document(
        filename, grobid_training_tei_structured_document):
    try:
        xml = etree.tostring(grobid_training_tei_structured_document.root)
    except Exception as e:
        raise RuntimeError('failed to convert to xml for %s due to %s' % (filename, e)) from e
    save_file_content(filename, xml)
