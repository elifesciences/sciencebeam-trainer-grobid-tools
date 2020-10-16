import logging
import os
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List, Union

from lxml import etree

from sciencebeam_utils.utils.xml import get_text_content

from sciencebeam_trainer_grobid_tools.utils.io import auto_download_input_file


LOGGER = logging.getLogger(__name__)


class XMLSyntaxErrorWithErrorLine(ValueError):
    def __init__(self, *args, error_line: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_line = error_line


def _read_lines(source) -> Iterable[str]:
    while True:
        line = source.readline()
        yield line


@contextmanager
def auto_download_and_fix_input_file(
        file_url_or_open_fn: Union[str, callable],
        fix_xml: bool = True) -> str:
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
                with BufferedReader(temp_fp) as reader:
                    skip_spaces(reader)
                    return etree.parse(reader, **kwargs)
        except etree.XMLSyntaxError as exception:
            error_lineno = exception.lineno
            with open(temp_file, mode='rb') as temp_fp:
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


def get_xpath_text(root: etree.Element, xpath: str, delimiter: str = ' ', **kwargs) -> str:
    return delimiter.join(get_text_content(node) for node in root.xpath(xpath, **kwargs))
