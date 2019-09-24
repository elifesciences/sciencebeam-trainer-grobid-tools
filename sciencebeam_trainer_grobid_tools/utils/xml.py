import logging
from functools import partial
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


def parse_xml_or_get_error_line(open_fn, filename: str = None, **kwargs):
    try:
        with open_fn() as source:
            return etree.parse(source, **kwargs)
    except etree.XMLSyntaxError as exception:
        error_lineno = exception.lineno
        with open_fn() as source:
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
