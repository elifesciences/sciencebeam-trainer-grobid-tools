from io import BytesIO
from pathlib import Path

from apache_beam.io.filesystems import FileSystems

from sciencebeam_trainer_grobid_tools.utils.xml import (
    parse_xml_or_get_error_line,
    XMLSyntaxErrorWithErrorLine
)


class TestParseXmlOrGetErrorLine:
    def test_should_parse_valid_xml(self):
        root = parse_xml_or_get_error_line(
            lambda: BytesIO(b'<xml>test</xml>')
        ).getroot()
        assert root.text == 'test'

    def test_should_parse_xml_with_space_infront_of_xml_declaration(self):
        root = parse_xml_or_get_error_line(
            lambda: BytesIO(b' <?xml version="1.0" encoding="UTF-8"?>\n<xml>test</xml>')
        ).getroot()
        assert root.text == 'test'

    def test_should_get_error_line(self):
        try:
            parse_xml_or_get_error_line(
                lambda: BytesIO(b'<xml>\n/xml>')
            )
            assert False
        except XMLSyntaxErrorWithErrorLine as e:
            assert e.error_line == b'/xml>'

    def test_should_get_error_line_using_beam_fs(self, temp_dir: Path):
        xml_file = temp_dir.joinpath('test.xml')
        xml_file.write_text('<xml>\n/xml>')
        try:
            parse_xml_or_get_error_line(
                lambda: FileSystems.open(str(xml_file))
            )
            assert False
        except XMLSyntaxErrorWithErrorLine as e:
            assert e.error_line == b'/xml>'

    def test_should_get_error_line_using_compressed_beam_fs(self, temp_dir: Path):
        xml_file = temp_dir.joinpath('test.xml.gz')
        with FileSystems.create(str(xml_file)) as fp:
            fp.write(b'<xml>\n/xml>')
        try:
            parse_xml_or_get_error_line(
                lambda: FileSystems.open(str(xml_file))
            )
            assert False
        except XMLSyntaxErrorWithErrorLine as e:
            assert e.error_line == b'/xml>'
