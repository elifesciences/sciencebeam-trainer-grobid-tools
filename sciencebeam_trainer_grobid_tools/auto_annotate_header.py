from __future__ import absolute_import

import argparse
import os
import logging

from lxml import etree

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.beam_utils.utils import (
    PreventFusion
)

from sciencebeam_utils.beam_utils.files import (
    FindFiles
)

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    parse_xml_mapping,
    xml_root_to_target_annotations
)


from sciencebeam_gym.preprocess.annotation.annotator import (
    Annotator,
    LineAnnotator
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
    MatchingAnnotator
)

from .utils.string import comma_separated_str_to_list

from .utils.regex import (
    regex_change_name
)

from .structured_document.annotator import annotate_structured_document


LOGGER = logging.getLogger(__name__)


def get_logger():
    return logging.getLogger(__name__)


class MetricCounters(object):
    FILE_PAIR = 'file_pair_count'
    PAGE = 'page_count'
    FILTERED_PAGE = 'filtered_page_count'
    CONVERT_PDF_TO_LXML_ERROR = 'ConvertPdfToLxml_error_count'
    CONVERT_PDF_TO_PNG_ERROR = 'ConvertPdfToPng_error_count'
    CONVERT_LXML_TO_SVG_ANNOT_ERROR = 'ConvertPdfToSvgAnnot_error_count'


def _file_exists(file_url):
    result = FileSystems.exists(file_url)
    LOGGER.debug('file exists: result=%s, url=%s', result, file_url)
    return result


def _get_xml_mapping_and_fields(xml_mapping_path, fields):
    xml_mapping = parse_xml_mapping(xml_mapping_path)
    if fields:
        xml_mapping = {
            top_level_key: {
                k: v
                for k, v in field_mapping.items()
                if k in fields
            }
            for top_level_key, field_mapping in xml_mapping.items()
        }
    else:
        fields = {
            k
            for top_level_key, field_mapping in xml_mapping.items()
            for k in field_mapping.items()
        }
    return xml_mapping, fields


def _load_xml(file_url):
    with FileSystems.open(file_url) as source_fp:
        return etree.parse(source_fp)


def _get_annotator(
        xml_path, xml_mapping, match_detail_reporter,
        use_tag_begin_prefix=False,
        use_line_no_annotator=False):

    annotators = []
    if use_line_no_annotator:
        annotators.append(LineAnnotator())
    if xml_path:
        target_annotations = xml_root_to_target_annotations(
            _load_xml(xml_path).getroot(),
            xml_mapping
        )
        annotators = annotators + [MatchingAnnotator(
            target_annotations, match_detail_reporter=match_detail_reporter,
            use_tag_begin_prefix=use_tag_begin_prefix
        )]
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(object):
    def __init__(self, opt):
        self.source_base_path = opt.source_base_path
        self.output_path = opt.output_path
        self.xml_path = opt.xml_path
        self.xml_filename_regex = opt.xml_filename_regex
        self.resume = opt.resume
        self.xml_mapping, self.fields = _get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields
        )
        self.preserve_tags = not opt.no_preserve_tags

    def get_tei_xml_output_file_for_source_file(self, source_url):
        return os.path.join(
            self.output_path,
            os.path.basename(source_url)
        )

    def output_file_not_exists(self, source_url):
        return not _file_exists(
            self.get_tei_xml_output_file_for_source_file(source_url)
        )

    def get_target_xml_for_source_file(self, source_url):
        return os.path.join(
            self.xml_path,
            regex_change_name(
                os.path.basename(source_url),
                self.xml_filename_regex
            )
        )

    def auto_annotate(self, source_url):
        try:
            output_xml_path = self.get_tei_xml_output_file_for_source_file(source_url)
            target_xml_path = self.get_target_xml_for_source_file(source_url)
            annotator = _get_annotator(
                target_xml_path,
                self.xml_mapping,
                match_detail_reporter=None
            )
            annotate_structured_document(
                source_url,
                output_xml_path,
                annotator=annotator,
                preserve_tags=self.preserve_tags,
                fields=self.fields
            )
        except Exception as e:
            get_logger().error('failed to process %s due to %s', source_url, e, exc_info=e)
            raise e

    def configure(self, p):
        tei_xml_file_url_source = FindFiles(os.path.join(
            self.source_base_path,
            '*.header.tei.xml*'
        ))

        tei_xml_input_urls = (
            p |
            tei_xml_file_url_source |
            PreventFusion()
        )

        if self.resume:
            tei_xml_input_urls |= "SkipAlreadyProcessed" >> beam.Filter(
                self.output_file_not_exists
            )

        _ = (
            tei_xml_input_urls |
            "Logging Input URI" >> beam.Map(lambda input_uri: get_logger().info(
                'input uri: %s', input_uri
            ))
        )

        _ = (
            tei_xml_input_urls |
            "Auto-Annotate" >> beam.Map(self.auto_annotate)
        )


def configure_pipeline(p, opt):
    return AnnotatePipelineFactory(opt).configure(p)


def add_main_args(parser):
    parser.add_argument(
        '--source-base-path', type=str, required=True,
        help='source training data path'
    )

    parser.add_argument(
        '--output-path', type=str, required=True,
        help='target training data path'
    )

    parser.add_argument(
        '--limit', type=int, required=False,
        help='limit the number of files to process'
    )

    parser.add_argument(
        '--xml-path', type=str, required=True,
        help='path to xml file(s)'
    )
    parser.add_argument(
        '--xml-filename-regex', type=str, required=True,
        help='regular expression to transform source filename to target xml filename'
    )
    parser.add_argument(
        '--xml-mapping-path', type=str, default='annot-xml-front.conf',
        help='path to xml mapping file'
    )
    parser.add_argument(
        '--fields',
        type=comma_separated_str_to_list,
        help='comma separated list of fields to annotate'
    )

    parser.add_argument(
        '--no-preserve-tags', action='store_true', required=False,
        help='do not preserve existing tags (tags other than the one being annotated)'
    )

    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='resume conversion (skip files that already have an output file)'
    )

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)
    add_cloud_args(parser)

    # parsed_args, other_args = parser.parse_known_args(argv)
    parsed_args = parser.parse_args(argv)

    process_cloud_args(
        parsed_args, parsed_args.output_path,
        name='sciencebeam-grobid-trainer-tools'
    )

    get_logger().info('parsed_args: %s', parsed_args)

    return parsed_args


def run(args: argparse.Namespace):
    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions.from_dictionary(vars(args))
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(args.runner, options=pipeline_options) as p:
        configure_pipeline(p, args)

        # Execute the pipeline and wait until it is completed.


def main(argv=None):
    args = parse_args(argv)

    if args.debug:
        logging.getLogger('sciencebeam_trainer_grobid_tools').setLevel('DEBUG')
        logging.getLogger('sciencebeam_gym').setLevel('DEBUG')

    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
