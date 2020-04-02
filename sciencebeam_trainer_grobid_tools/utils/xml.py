import logging
from contextlib import contextmanager
from functools import partial
from io import BufferedReader
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Iterable

from lxml import etree

from apache_beam.io.filesystems import FileSystems


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
def stream_to_local_file(open_fn: callable):
    with open_fn() as source:
        with NamedTemporaryFile() as temp_file:
            copyfileobj(source, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            yield temp_file


def skip_spaces(reader: BufferedReader):
    while True:
        peeked_value = reader.peek(1)[:1]
        if peeked_value != b' ':
            return
        LOGGER.debug('skipping character: %s', peeked_value)
        reader.read(1)


def parse_xml_or_get_error_line(open_fn, filename: str = None, **kwargs):
    with stream_to_local_file(open_fn) as temp_file:
        try:
            with open(temp_file.name, mode='rb') as source:
                with BufferedReader(source) as reader:
                    skip_spaces(reader)
                    return etree.parse(reader, **kwargs)
        except etree.XMLSyntaxError as exception:
            error_lineno = exception.lineno
            with open(temp_file.name, mode='rb') as source:
                try:
                    line_enumeration = enumerate(source, 1)
                except TypeError:
                    line_enumeration = enumerate(_read_lines(source), 1)
                for current_lineno, line in line_enumeration:
                    if current_lineno == error_lineno:
                        raise XMLSyntaxErrorWithErrorLine(
                            f'failed to parse xml file "%s", line=[{line}] due to {exception}' % (
                                filename or exception.filename
                            ),
                            error_line=line
                        ) from exception
            raise exception


def parse_xml(file_url):
    return parse_xml_or_get_error_line(
        partial(FileSystems.open, file_url),
        filename=file_url
    )
