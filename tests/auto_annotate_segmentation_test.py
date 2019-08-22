import logging

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_grobid_tools.auto_annotate_segmentation import (
    main
)

from .test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)


XML_FILENAME_1 = 'document1.xml'
TEI_FILENAME_1 = 'document1.segmentation.tei.xml'

TEI_FILENAME_REGEX = r'/(.*).segmentation.tei.xml/\1.xml/'

TOKEN_1 = 'token1'


class TestEndToEnd(object):
    @log_on_exception
    def test_should_auto_annotate_title_as_front(self, temp_dir):
        tei_raw_path = temp_dir.joinpath('tei-raw')
        tei_auto_path = temp_dir.joinpath('tei-auto')
        xml_path = temp_dir.joinpath('xml')
        tei_raw_path.mkdir()
        tei_raw_path.joinpath(TEI_FILENAME_1).write_bytes(etree.tostring(
            E.tei(E.text(
                E.note(TOKEN_1)
            ))
        ))
        xml_path.mkdir()
        xml_path.joinpath(XML_FILENAME_1).write_bytes(etree.tostring(
            E.article(E.front(
                E('article-meta', E('title-group', E('article-title', TOKEN_1)))
            ))
        ))
        main([
            '--source-base-path=%s' % tei_raw_path,
            '--output-path=%s' % tei_auto_path,
            '--xml-path=%s' % xml_path,
            '--xml-filename-regex=%s' % TEI_FILENAME_REGEX,
            '--fields=title,abstract'
        ], save_main_session=False)
        tei_auto_file_path = tei_auto_path.joinpath(TEI_FILENAME_1)
        assert tei_auto_file_path.exists()
        tei_auto_root = etree.parse(str(tei_auto_file_path)).getroot()
        LOGGER.info('tei_auto_root: %s', etree.tostring(tei_auto_root))
        front_nodes = tei_auto_root.xpath('//text/front')
        assert front_nodes
        assert front_nodes[0].text == TOKEN_1
