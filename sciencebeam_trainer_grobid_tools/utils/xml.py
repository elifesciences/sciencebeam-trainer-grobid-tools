import logging
import os
import xml.sax.saxutils
from contextlib import contextmanager
from html.parser import HTMLParser
from io import BufferedReader, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator, List, Tuple, Union

from lxml import etree

from sciencebeam_utils.beam_utils.io import read_all_from_path, save_file_content

from sciencebeam_utils.utils.xml import get_text_content_list

from sciencebeam_trainer_grobid_tools.utils.io import (
    T_BinaryIO_Open_Function,
    auto_download_input_file
)


LOGGER = logging.getLogger(__name__)


def iter_text_content_and_exclude(
        node: etree.Element,
        exclude: List[etree.Element] = None) -> Iterable[str]:
    if not exclude:
        yield from node.itertext()
        return
    if node.text is not None:
        yield node.text
    for child in node.iterchildren():
        if child not in exclude:
            yield from iter_text_content_and_exclude(child, exclude=exclude)
        if child.tail:
            yield child.tail


class XMLSyntaxErrorWithErrorLine(ValueError):
    def __init__(self, *args, error_line: Union[bytes, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.error_line = error_line


def _read_lines(source) -> Iterable[str]:
    while True:
        line = source.readline()
        yield line


@contextmanager
def auto_download_and_fix_input_file(
    file_url_or_open_fn: Union[str, T_BinaryIO_Open_Function],
    fix_xml: bool = True
) -> Iterator[str]:
    with auto_download_input_file(file_url_or_open_fn) as temp_file:
        if not fix_xml:
            yield temp_file
            return
        with TemporaryDirectory(suffix='-input-fixed') as fixed_temp_dir:
            fixed_temp_file = os.path.join(
                fixed_temp_dir, os.path.basename(temp_file)
            )
            data = Path(temp_file).read_bytes()
            data = data.lstrip()
            data = data.replace(b'&dagger;', b'&#x2020;')
            Path(fixed_temp_file).write_bytes(data)
            yield fixed_temp_file


def skip_spaces(reader: BufferedReader):
    while True:
        peeked_value = reader.peek(1)[:1]
        if peeked_value != b' ':
            return
        LOGGER.debug('skipping character: %s', peeked_value)
        reader.read(1)


def parse_xml_or_get_error_line(
        source,
        filename: str = None,
        fix_xml: bool = False,
        **kwargs):
    with auto_download_and_fix_input_file(source, fix_xml=fix_xml) as temp_file:
        try:
            with open(temp_file, mode='rb') as temp_fp:
                # mypy: https://github.bajins.com/python/mypy/issues/10271
                with BufferedReader(temp_fp) as reader:  # type: ignore
                    skip_spaces(reader)
                    return etree.parse(reader, **kwargs)
        except etree.XMLSyntaxError as exception:
            error_lineno = exception.lineno
            with open(temp_file, mode='rb') as temp_fp:
                line_enumeration: Iterable[Tuple[int, Union[bytes, str]]]
                try:
                    line_enumeration = enumerate(temp_fp, 1)
                except TypeError:
                    line_enumeration = enumerate(_read_lines(temp_fp), 1)
                for current_lineno, line in line_enumeration:
                    if current_lineno == error_lineno:
                        raise XMLSyntaxErrorWithErrorLine(
                            'failed to parse xml file "%s", line=%r due to %r' % (
                                filename or exception.filename,
                                line,
                                exception
                            ),
                            error_line=line
                        ) from exception
            raise exception


def parse_xml(file_url: str, **kwargs) -> etree.ElementTree:
    return parse_xml_or_get_error_line(
        file_url,
        **kwargs
    )


def get_xpath_matches(
        parent: etree.Element,
        xpath: str,
        required: bool = False,
        **kwargs) -> List[etree.Element]:
    result = parent.xpath(xpath, **kwargs)
    if required and not result:
        xpath_fragments = xpath.split('/')
        for fragment_count in reversed(range(1, len(xpath_fragments))):
            parent_xpath = '/'.join(xpath_fragments[:fragment_count])
            if len(parent_xpath) <= 1:
                break
            parent_result = parent.xpath(parent_xpath, **kwargs)
            if parent_result:
                raise ValueError(
                    (
                        'no item found for xpath: %r (in %s),'
                        ' but found matching elements for %r: %s'
                    ) % (
                        xpath, parent, parent_xpath, parent_result
                    )
                )
        raise ValueError('no item found for xpath: %r (in %s)' % (xpath, parent))
    return result


def get_first_xpath_match(
        parent: etree.Element,
        xpath: str,
        **kwargs) -> etree.Element:
    return get_xpath_matches(parent, xpath, required=True, **kwargs)[0]


def get_xpath_text_list(root: etree.Element, xpath: str, **kwargs) -> str:
    return get_text_content_list(root.xpath(xpath, **kwargs))


def get_xpath_text(root: etree.Element, xpath: str, delimiter: str = ' ', **kwargs) -> str:
    return delimiter.join(get_xpath_text_list(root, xpath, **kwargs))


class FixingHtmlParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = StringIO()
        self.tag_stack = []

    def error(self, message):
        raise RuntimeError('parser error: %r' % message)

    def handle_starttag(self, tag, attrs):
        LOGGER.debug('starttag: tag=%r, attrs=%s', tag, attrs)
        self.buffer.write(f'<{tag}')
        for key, value in attrs:
            quoted_value = xml.sax.saxutils.quoteattr(value)
            self.buffer.write(f' {key}={quoted_value}')
        self.buffer.write('>')
        self.tag_stack.append(tag)

    def _close_current_element(self):
        tag = self.tag_stack.pop()
        LOGGER.debug('endtag: tag=%r', tag)
        self.buffer.write(f'</{tag}>')

    @property
    def _current_tag(self):
        return self.tag_stack[-1]

    def handle_endtag(self, tag):
        if not self.tag_stack:
            LOGGER.warning('attempting to close element without any open elements, tag=%r', tag)
            return
        if tag == self._current_tag:
            self._close_current_element()
            return
        if tag not in self.tag_stack:
            LOGGER.warning(
                'end tag tag=%r not matching any open element, closing tag=%r',
                tag, self._current_tag
            )
            self._close_current_element()
            return
        while tag != self._current_tag:
            LOGGER.warning(
                'end tag tag=%r not matching immediate open element, first closing tag=%r',
                tag, self._current_tag
            )
            self._close_current_element()

    def handle_data(self, data):
        LOGGER.debug('endtag: data=%r', data)
        self.buffer.write(xml.sax.saxutils.escape(data))

    def close(self) -> None:
        super().close()
        while self.tag_stack:
            self._close_current_element()


def get_fixed_xml_str(xml_str: str) -> str:
    try:
        # return passed in xml if it is valid
        etree.fromstring(xml_str)
        LOGGER.debug('passed in xml is valid, returning as is')
        return xml_str
    except etree.XMLSyntaxError:
        pass
    p = FixingHtmlParser()
    p.feed(xml_str)
    p.close()
    fixed_xml_str = p.buffer.getvalue()
    LOGGER.debug('fixed xml: %r', fixed_xml_str)
    return fixed_xml_str


def get_fixed_xml_bytes(xml_data: bytes, encoding: str = 'utf-8') -> bytes:
    return get_fixed_xml_str(xml_data.decode(encoding)).encode(encoding)


def fix_source_file_to(source_url: str, target_url: str, encoding: str = 'utf-8'):
    xml_bytes = read_all_from_path(source_url)
    fixed_xml_bytes = get_fixed_xml_bytes(xml_bytes, encoding=encoding)
    save_file_content(target_url, fixed_xml_bytes)


@contextmanager
def get_fixed_source_url(source_url: str) -> Iterator[str]:
    with TemporaryDirectory(suffix='-fixed') as temp_dir:
        fixed_source_url = os.path.join(temp_dir, os.path.basename(source_url))
        fix_source_file_to(source_url, fixed_source_url)
        yield fixed_source_url
