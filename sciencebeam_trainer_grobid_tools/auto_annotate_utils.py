import argparse
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Set

from lxml import etree

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_utils.beam_utils.utils import PreventFusion
from sciencebeam_utils.beam_utils.files import FindFiles

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    xml_root_to_target_annotations
)

from sciencebeam_gym.preprocess.annotation.annotator import LineAnnotator
from sciencebeam_gym.preprocess.annotation.matching_annotator import MatchingAnnotator

from sciencebeam_gym.preprocess.annotation.annotator import AbstractAnnotator
from sciencebeam_gym.preprocess.annotation.target_annotation import (
    parse_xml_mapping
)

from .utils.regex import regex_change_name
from .structured_document.annotator import annotate_structured_document


LOGGER = logging.getLogger(__name__)


def get_logger():
    return logging.getLogger(__name__)


def add_debug_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )
    return parser


def process_debug_argument(args: argparse.Namespace):
    if args.debug:
        logging.getLogger('sciencebeam_trainer_grobid_tools').setLevel('DEBUG')
        logging.getLogger('sciencebeam_gym').setLevel('DEBUG')


def get_default_config_path(filename: str):
    return os.path.join('config', filename)


def add_annotation_pipeline_arguments(parser: argparse.ArgumentParser):
    source_group = parser.add_argument_group('source')
    source_group.add_argument(
        '--source-base-path', type=str,
        help='source base data path for files to auto-annotate'
    )
    source_group.add_argument(
        '--source-path', type=str,
        help='source path to a specific file to auto-annotate'
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
        '--xml-mapping-path', type=str,
        default=get_default_config_path('annot-xml-front.conf'),
        help='path to xml mapping file'
    )

    parser.add_argument(
        '--no-preserve-tags', action='store_true', required=False,
        help='do not preserve existing tags (tags other than the one being annotated)'
    )

    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='resume conversion (skip files that already have an output file)'
    )
    add_cloud_args(parser)
    return parser


def process_annotation_pipeline_arguments(
        parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not (args.source_base_path or args.source_path):
        parser.error('one of --source-base-path or --source-path required')
    process_cloud_args(
        args, args.output_path,
        name='sciencebeam-grobid-trainer-tools'
    )


def _file_exists(file_url):
    result = FileSystems.exists(file_url)
    LOGGER.debug('file exists: result=%s, url=%s', result, file_url)
    return result


def load_xml(file_url):
    with FileSystems.open(file_url) as source_fp:
        return etree.parse(source_fp)


def get_xml_mapping_and_fields(xml_mapping_path, fields):
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


def get_default_annotators(
        xml_path, xml_mapping, match_detail_reporter,
        use_tag_begin_prefix=False,
        use_line_no_annotator=False) -> List[AbstractAnnotator]:

    annotators = []
    if use_line_no_annotator:
        annotators.append(LineAnnotator())
    if xml_path:
        target_annotations = xml_root_to_target_annotations(
            load_xml(xml_path).getroot(),
            xml_mapping
        )
        annotators = annotators + [MatchingAnnotator(
            target_annotations, match_detail_reporter=match_detail_reporter,
            use_tag_begin_prefix=use_tag_begin_prefix
        )]
    return annotators


class AbstractAnnotatePipelineFactory(ABC):
    def __init__(
            self,
            opt: argparse.Namespace,
            tei_filename_pattern: str,
            container_node_path: str,
            tag_to_tei_path_mapping: Dict[str, str] = None,
            output_fields: Set[str] = None):
        self.tei_filename_pattern = tei_filename_pattern
        self.container_node_path = container_node_path
        self.tag_to_tei_path_mapping = tag_to_tei_path_mapping
        self.source_base_path = opt.source_base_path
        self.source_path = opt.source_path
        self.output_path = opt.output_path
        self.xml_path = opt.xml_path
        self.xml_filename_regex = opt.xml_filename_regex
        self.limit = opt.limit
        self.resume = opt.resume
        self.preserve_tags = not opt.no_preserve_tags
        self.output_fields = output_fields

    @abstractmethod
    def get_annotator(self, source_url: str):
        pass

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

    def auto_annotate(self, source_url: str):
        try:
            output_xml_path = self.get_tei_xml_output_file_for_source_file(source_url)
            annotator = self.get_annotator(source_url)
            annotate_structured_document(
                source_url,
                output_xml_path,
                annotator=annotator,
                preserve_tags=self.preserve_tags,
                fields=self.output_fields,
                container_node_path=self.container_node_path,
                tag_to_tei_path_mapping=self.tag_to_tei_path_mapping
            )
        except Exception as e:
            get_logger().error('failed to process %s due to %s', source_url, e, exc_info=e)
            raise e

    def run(self, args: argparse.Namespace, save_main_session: bool = True):
        # We use the save_main_session option because one or more DoFn's in this
        # workflow rely on global context (e.g., a module imported at module level).
        pipeline_options = PipelineOptions.from_dictionary(vars(args))
        pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

        with beam.Pipeline(args.runner, options=pipeline_options) as p:
            self.configure(p)

            # Execute the pipeline and wait until it is completed.

    def get_source_files(self):
        if self.source_path:
            return beam.Create([self.source_path])
        return FindFiles(os.path.join(
            self.source_base_path,
            self.tei_filename_pattern
        ), limit=self.limit)

    def configure(self, p):
        tei_xml_file_url_source = self.get_source_files()

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
