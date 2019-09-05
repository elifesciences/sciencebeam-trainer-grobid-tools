import argparse
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Set

from lxml import etree

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_utils.utils.csv import open_csv_output

from sciencebeam_utils.beam_utils.files import find_matching_filenames_with_limit
from sciencebeam_utils.tools.check_file_list import map_file_list_to_file_exists

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    xml_root_to_target_annotations
)

from sciencebeam_gym.preprocess.annotation.annotator import LineAnnotator
from sciencebeam_gym.preprocess.annotation.matching_annotator import (
    get_simple_fuzzy_match_filter,
    MatchingAnnotatorConfig,
    MatchingAnnotator,
    CsvMatchDetailReporter,
    DEFAULT_SEQ_MIN_MATCH_COUNT,
    DEFAULT_SEQ_RATIO_MIN_MATCH_COUNT,
    DEFAULT_CHOICE_MIN_MATCH_COUNT,
    DEFAULT_CHOICE_RATIO_MIN_MATCH_COUNT
)

from sciencebeam_gym.preprocess.annotation.annotator import AbstractAnnotator
from sciencebeam_gym.preprocess.annotation.target_annotation import (
    parse_xml_mapping
)

from .utils.string import comma_separated_str_to_list
from .utils.regex import regex_change_name
from .structured_document.annotator import annotate_structured_document
from .structured_document.simple_matching_annotator import (
    SimpleMatchingAnnotator,
    SimpleSimpleMatchingConfig
)


LOGGER = logging.getLogger(__name__)


def get_logger():
    return logging.getLogger(__name__)


class MatcherNames:
    DEFAULT = 'default'
    SIMPLE = 'simple'


MATCHER_NAMES = [MatcherNames.DEFAULT, MatcherNames.SIMPLE]


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
        '--always-preserve-fields',
        type=comma_separated_str_to_list,
        help='always preserve the listed fields (they will be excluded from the matcher)'
    )

    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='resume conversion (skip files that already have an output file)'
    )

    matcher_group = parser.add_argument_group('matcher')
    matcher_group.add_argument(
        '--matcher', type=str, choices=MATCHER_NAMES,
        default=MatcherNames.DEFAULT,
        help=''.join([
            'the kind of matcher to use ("simple" uses a simpler algorith,',
            ' requiring correct reading order)'
        ])
    )
    matcher_group.add_argument(
        '--matcher-score-threshold', type=float, default=0.8,
        help='score threshold for a match to be accepted (1.0 for exact match)'
    )
    matcher_group.add_argument(
        '--matcher-lookahead-lines', type=int, default=500,
        help='simple matcher only: number of lines to try to find matches for'
    )
    matcher_group.add_argument(
        '--debug-match', type=str, required=False,
        help='if set, path to csv or tsv file with debug matches'
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


def get_match_detail_reporter(debug_match: str) -> CsvMatchDetailReporter:
    if not debug_match:
        return None
    return CsvMatchDetailReporter(
        open_csv_output(debug_match),
        debug_match
    )


class AnnotatorConfig:
    def __init__(
            self,
            matcher_name: str,
            score_threshold: float,
            lookahead_lines: int,
            debug_match: str = None):
        self.matcher_name = matcher_name
        self.score_threshold = score_threshold
        self.lookahead_lines = lookahead_lines
        self.debug_match = debug_match

    def get_match_detail_reporter(self):
        return get_match_detail_reporter(self.debug_match)

    def get_simple_annotator_config(self) -> SimpleSimpleMatchingConfig:
        return SimpleSimpleMatchingConfig(
            threshold=self.score_threshold,
            lookahead_sequence_count=self.lookahead_lines
        )

    def get_matching_annotator_config(self) -> MatchingAnnotatorConfig:
        return MatchingAnnotatorConfig(
            match_detail_reporter=self.get_match_detail_reporter(),
            seq_match_filter=get_simple_fuzzy_match_filter(
                score_threshold=self.score_threshold,
                min_match_count=DEFAULT_SEQ_MIN_MATCH_COUNT,
                ratio_min_match_count=DEFAULT_SEQ_RATIO_MIN_MATCH_COUNT
            ),
            choice_match_filter=get_simple_fuzzy_match_filter(
                score_threshold=self.score_threshold,
                min_match_count=DEFAULT_CHOICE_MIN_MATCH_COUNT,
                ratio_min_match_count=DEFAULT_CHOICE_RATIO_MIN_MATCH_COUNT
            )
        )


def get_default_annotators(
        xml_path, xml_mapping,
        annotator_config: AnnotatorConfig,
        use_line_no_annotator=False) -> List[AbstractAnnotator]:

    annotators = []
    if use_line_no_annotator:
        annotators.append(LineAnnotator())
    if xml_path:
        target_annotations = xml_root_to_target_annotations(
            load_xml(xml_path).getroot(),
            xml_mapping
        )
        if annotator_config.matcher_name == MatcherNames.SIMPLE:
            annotators.append(SimpleMatchingAnnotator(
                target_annotations,
                config=annotator_config.get_simple_annotator_config()
            ))
        else:
            annotators.append(MatchingAnnotator(
                target_annotations,
                matching_annotator_config=annotator_config.get_matching_annotator_config()
            ))
    return annotators


def get_file_list_without_output_file(
        file_list: List[str],
        get_output_file_for_source_url: Callable[[str], str]) -> List[str]:
    output_file_exists_list = map_file_list_to_file_exists([
        get_output_file_for_source_url(file_url)
        for file_url in file_list
    ])
    LOGGER.debug('output_file_exists_list: %s', output_file_exists_list)
    return [
        file_url
        for file_url, output_file_exists in zip(file_list, output_file_exists_list)
        if not output_file_exists
    ]


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
        self.always_preserve_fields = opt.always_preserve_fields
        self.output_fields = output_fields
        self.annotator_config = AnnotatorConfig(
            matcher_name=opt.matcher,
            score_threshold=opt.matcher_score_threshold,
            lookahead_lines=opt.matcher_lookahead_lines,
            debug_match=opt.debug_match
        )

    @abstractmethod
    def get_annotator(self, source_url: str):
        pass

    def get_annotator_config(self) -> AnnotatorConfig:
        return self.annotator_config

    def get_tei_xml_output_file_for_source_file(self, source_url):
        return os.path.join(
            self.output_path,
            os.path.basename(source_url)
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
                always_preserve_fields=self.always_preserve_fields,
                container_node_path=self.container_node_path,
                tag_to_tei_path_mapping=self.tag_to_tei_path_mapping
            )
        except Exception as e:
            get_logger().error('failed to process %s due to %s', source_url, e, exc_info=e)
            raise e

    def get_source_file_list(self):
        if self.source_path:
            return [self.source_path]
        return list(find_matching_filenames_with_limit(os.path.join(
            self.source_base_path,
            self.tei_filename_pattern
        ), limit=self.limit))

    def get_remaining_source_file_list(self):
        file_list = self.get_source_file_list()
        LOGGER.debug('file_list: %s', file_list)

        if not file_list:
            LOGGER.warning('no files found')
            return file_list

        LOGGER.info('total number of files: %d', len(file_list))
        if self.resume:
            file_list = get_file_list_without_output_file(
                file_list,
                get_output_file_for_source_url=self.get_tei_xml_output_file_for_source_file
            )
            LOGGER.info('remaining number of files: %d', len(file_list))
        return file_list

    def configure_beam_pipeline(self, p: beam.Pipeline):
        tei_xml_file_list = self.get_remaining_source_file_list()

        tei_xml_input_urls = (
            p |
            beam.Create(tei_xml_file_list)
        )

        _ = (
            tei_xml_input_urls |
            "Auto-Annotate" >> beam.Map(self.auto_annotate)
        )

    def run_beam_pipeline(self, args: argparse.Namespace, save_main_session: bool = True):
        # We use the save_main_session option because one or more DoFn's in this
        # workflow rely on global context (e.g., a module imported at module level).
        pipeline_options = PipelineOptions.from_dictionary(vars(args))
        pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

        with beam.Pipeline(args.runner, options=pipeline_options) as p:
            self.configure_beam_pipeline(p)

            # Execute the pipeline and wait until it is completed.

    def run(self, args: argparse.Namespace, save_main_session: bool = True):
        self.run_beam_pipeline(args, save_main_session=save_main_session)
