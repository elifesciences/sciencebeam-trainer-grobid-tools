from io import BytesIO
from pathlib import Path

from apache_beam.io.filesystems import FileSystems

from sciencebeam_trainer_grobid_tools.utils.xml import (
    parse_xml_or_get_error_line,
    XMLSyntaxErrorWithErrorLine,
    get_fixed_xml_str
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


class TestGetFixedXmlStr:
    def test_should_preserve_valid_xml_without_ns(self):
        xml_str = '<tei><text>abc</text></tei>'
        assert get_fixed_xml_str(xml_str) == xml_str

    def test_should_preserve_valid_xml_with_ns(self):
        xml_str = '<tei xmlns="http://www.tei-c.org/ns/1.0"><text>abc</text></tei>'
        assert get_fixed_xml_str(xml_str) == xml_str

    def test_should_correct_single_closing_tag(self):
        xml_str = (
            '<tei><text><figure>abc</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure>abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_multiple_single_closing_tag(self):
        xml_str = (
            '<tei><text><figure>abc</p><figure>abc</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure>abc</figure><figure>abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_close_unclosed_tag(self):
        xml_str = (
            '<tei><text><figure>abc</text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure>abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_closing_tag_with_ns(self):
        xml_str = (
            '<tei xmlns="http://www.tei-c.org/ns/1.0"><text><figure>abc</p></text></tei>'
        )
        expected_xml_str = (
            '<tei xmlns="http://www.tei-c.org/ns/1.0"><text><figure>abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_closing_tag_with_attributes(self):
        xml_str = (
            '<tei><text><figure a="1" b="2">abc</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure a="1" b="2">abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_closing_tag_and_escape_gt_in_data(self):
        xml_str = (
            '<tei><text><figure a="1" b="2">a &gt; b</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure a="1" b="2">a &gt; b</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_closing_tag_and_decode_apos_in_data(self):
        xml_str = (
            '<tei><text><figure a="1" b="2">a &apos; b</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure a="1" b="2">a \' b</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str

    def test_should_correct_closing_tag_and_escape_amp_in_attrib(self):
        xml_str = (
            '<tei><text><figure a="1 &amp; 2">abc</p></text></tei>'
        )
        expected_xml_str = (
            '<tei><text><figure a="1 &amp; 2">abc</figure></text></tei>'
        )
        assert get_fixed_xml_str(xml_str) == expected_xml_str
