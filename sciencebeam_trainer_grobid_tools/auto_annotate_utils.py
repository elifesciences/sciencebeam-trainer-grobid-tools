import argparse
import logging
import concurrent.futures
import re
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_utils.utils.csv import open_csv_output
from sciencebeam_utils.utils.progress_logger import logging_tqdm
from sciencebeam_utils.beam_utils.files import find_matching_filenames_with_limit
from sciencebeam_utils.tools.check_file_list import map_file_list_to_file_exists

from .core.annotation.matching_annotator import (
    get_simple_fuzzy_match_filter,
    MatchingAnnotatorConfig,
    MatchingAnnotator,
    CsvMatchDetailReporter,
    DEFAULT_SEQ_MIN_MATCH_COUNT,
    DEFAULT_SEQ_RATIO_MIN_MATCH_COUNT,
    DEFAULT_CHOICE_MIN_MATCH_COUNT,
    DEFAULT_CHOICE_RATIO_MIN_MATCH_COUNT
)

from .core.annotation.annotator import (
    AbstractAnnotator
)

from .core.annotation.target_annotation import (
    XmlMappingSuffix,
    parse_xml_mapping
)

from .annotation.target_annotation import (
    xml_root_to_target_annotations
)

from .utils.string import (
    comma_separated_str_to_list,
    get_plus_minus_comma_separated_str_to_list_fn,
    parse_dict
)
from .utils.regex import regex_change_name
from .utils.xml import parse_xml
from .annotation.annotator import annotate_structured_document
from .annotation.checks import (
    is_structured_document_passing_checks,
    get_target_annotations_from_annotator
)
from .annotation.target_annotation import TargetAnnotation

from .structured_document.grobid_training_tei import GrobidTrainingTeiStructuredDocument

from .annotation.line_number_annotator import (
    DEFAULT_MIN_LINE_NUMBER_COUNT,
    DEFAULT_MAX_LINE_NUMBER_GAP,
    DEFAULT_LINE_NUMBER_RATIO_THRESHOLD,
    TextLineNumberAnnotatorConfig,
    TextLineNumberAnnotator
)

from .annotation.simple_matching_annotator import (
    SimpleMatchingAnnotator,
    SimpleSimpleMatchingConfig,
    get_simple_tag_config_map
)


LOGGER = logging.getLogger(__name__)


def get_logger():
    return logging.getLogger(__name__)


class MatcherNames:
    COMPLEX = 'complex'
    SIMPLE = 'simple'


MATCHER_NAMES = [MatcherNames.COMPLEX, MatcherNames.SIMPLE]

DEFAULT_MATCHER_NAME = MatcherNames.SIMPLE

DEFAULT_ANNOT_CONFIG_FILENAME = 'xml-mapping.conf'


def add_debug_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )
    return parser


def process_debug_argument(args: argparse.Namespace):
    if args.debug:
        for name in {'sciencebeam_trainer_grobid_tools', '__main__'}:
            logging.getLogger(name).setLevel('DEBUG')


def get_default_config_path(filename: str):
    return os.path.join('config', filename)


def add_annotation_pipeline_arguments(
        parser: argparse.ArgumentParser,
        default_matcher_lookahead_lines: int = 500):
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
        '--failed-output-path', type=str, required=False,
        help=(
            'Target data path where documents should be saved to, if they fail quality checks.'
            ' Leave blank if those documents should not be saved.'
        )
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
        default=get_default_config_path(DEFAULT_ANNOT_CONFIG_FILENAME),
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
        default=DEFAULT_MATCHER_NAME,
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
        '--matcher-lookahead-lines',
        type=int,
        default=default_matcher_lookahead_lines,
        help='simple matcher only: number of lines to try to find matches for'
    )
    matcher_group.add_argument(
        '--debug-match', type=str, required=False,
        help='if set, path to csv or tsv file with debug matches'
    )

    parser.add_argument(
        '--multi-processing', action='store_true', default=False,
        help='enable multi processing rather than multi threading'
    )

    parser.add_argument(
        '--skip-errors', action='store_true', default=False,
        help='skip errors'
    )

    line_no_group = parser.add_argument_group('line number annotation')
    line_no_group.add_argument(
        '--use-line-number-annotator',
        dest='use_line_number_annotator',
        action='store_true',
        default=False,
        help='Enable line number annotator'
    )
    line_no_group.add_argument(
        '--no-line-number-annotator',
        dest='use_line_number_annotator',
        action='store_false',
        default=False,
        help='Disable line number annotator'
    )
    line_no_group.add_argument(
        '--min-line-numbers-per-page', type=int,
        default=DEFAULT_MIN_LINE_NUMBER_COUNT,
        help='minimum number of line number candidates on page to be considered'
    )
    line_no_group.add_argument(
        '--max-line-number-gap', type=int,
        default=DEFAULT_MAX_LINE_NUMBER_GAP,
        help=' '.join([
            'the maximum interval gap between line numbers',
            '(some documents only show line numbers on lines with text)'
        ])
    )
    line_no_group.add_argument(
        '--min-line-number-ratio', type=str,
        default=DEFAULT_LINE_NUMBER_RATIO_THRESHOLD,
        help=' '.join([
            'minimum ratio of line number candidates vs non-line number tokens',
            ' (first token of line)'
        ])
    )

    line_no_group.add_argument(
        '--xml-mapping-overrides', type=parse_dict,
        help=' '.join([
            'override xml mapping values, in the format: key1=value1|key2=value2'
        ])
    )

    add_cloud_args(parser)
    return parser


def add_document_checks_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--require-matching-fields',
        type=comma_separated_str_to_list,
        help=(
            'Comma separated list of fields that are required to match (if present).'
            ' XML files are discarded if one of those fields do not meet the threshold.'
        )
    )
    parser.add_argument(
        '--required-fields',
        type=comma_separated_str_to_list,
        help=(
            'Comma separated list of fields that are required to be present.'
            ' Where the target value is missing, this would cause the document to fail.'
        )
    )


def add_fields_argument(
        parser: argparse.ArgumentParser,
        default_fields: List[str] = None):
    parser.add_argument(
        '--fields',
        type=get_plus_minus_comma_separated_str_to_list_fn(default_fields),
        default=','.join(default_fields) if default_fields else None,
        help='comma separated list of fields to annotate'
    )


def add_sub_fields_argument(
        parser: argparse.ArgumentParser,
        default_sub_fields: List[str] = None):
    parser.add_argument(
        '--sub-fields',
        type=get_plus_minus_comma_separated_str_to_list_fn(default_sub_fields),
        default=','.join(default_sub_fields) if default_sub_fields else None,
        help=(
            'comma separated list of sub fields to annotate.'
            ' if blank, all available sub fields will be used.'
        )
    )


def add_preserve_sub_tags_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--preserve-sub-tags',
        action='store_true',
        default=False,
        help='enable preserving sub tags.'
    )


def add_no_preserve_sub_fields_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--no-preserve-sub-fields',
        type=comma_separated_str_to_list,
        help='comma separated list of sub fields that should not be preserved'
    )


def process_annotation_pipeline_arguments(
        parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not (args.source_base_path or args.source_path):
        parser.error('one of --source-base-path or --source-path required')
    process_cloud_args(
        args, args.output_path,
        name='sciencebeam-grobid-trainer-tools'
    )


def get_mapping_key_field_name(key: str) -> str:
    return key.split('.', maxsplit=1)[0]


def get_sub_field_for_key_or_none(key: str) -> Optional[str]:
    field_name = get_mapping_key_field_name(key)
    sub_prefix = field_name + XmlMappingSuffix.SUB + '.'
    if not key.startswith(sub_prefix):
        return None
    return key[len(sub_prefix):]


def is_key_selected_sub_fields_or_no_sub_field(
        key: str,
        sub_fields: List[str]) -> bool:
    sub_field = get_sub_field_for_key_or_none(key)
    if not sub_field:
        return True
    return sub_field in sub_fields


def get_xml_mapping_with_filtered_sub_fields(
        xml_mapping: Dict[str, Dict[str, str]],
        sub_fields: List[str] = None) -> Dict[str, Dict[str, str]]:
    if not sub_fields:
        return xml_mapping
    LOGGER.debug('selecting sub_fields: %s', sub_fields)
    return {
        top_level_key: {
            k: v
            for k, v in field_mapping.items()
            if is_key_selected_sub_fields_or_no_sub_field(k, sub_fields)
        }
        for top_level_key, field_mapping in xml_mapping.items()
    }


def get_filtered_xml_mapping_and_fields(
    xml_mapping: Dict[str, Dict[str, str]],
    fields: Optional[Set[str]],
    sub_fields: Optional[Set[str]] = None
) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    if fields:
        result_fields=fields
        xml_mapping = {
            top_level_key: {
                k: v
                for k, v in field_mapping.items()
                if get_mapping_key_field_name(k) in fields
            }
            for top_level_key, field_mapping in xml_mapping.items()
        }
    else:
        result_fields = {
            k
            for top_level_key, field_mapping in xml_mapping.items()
            for k in field_mapping.keys()
            if '.' not in k
        }
    xml_mapping = get_xml_mapping_with_filtered_sub_fields(
        xml_mapping,
        sub_fields=sub_fields
    )
    return xml_mapping, result_fields


def get_xml_mapping_with_overrides(
        xml_mapping: Dict[str, Dict[str, str]],
        xml_mapping_overrides: Dict[str, str]):
    if not xml_mapping_overrides:
        return xml_mapping
    return {
        top_level_key: {
            **field_mapping,
            **xml_mapping_overrides
        }
        for top_level_key, field_mapping in xml_mapping.items()
    }


def get_xml_mapping_and_fields(
    xml_mapping_path: str,
    fields: Optional[Set[str]],
    sub_fields: List[str] = None,
    xml_mapping_overrides: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    return get_filtered_xml_mapping_and_fields(
        get_xml_mapping_with_overrides(
            parse_xml_mapping(xml_mapping_path),
            xml_mapping_overrides=xml_mapping_overrides
        ),
        fields,
        sub_fields=sub_fields
    )


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
            debug_match: str = None,
            use_line_number_annotator: bool = False,
            use_sub_annotations: bool = False,
            line_number_annotator_config: TextLineNumberAnnotatorConfig = None):
        self.matcher_name = matcher_name
        self.score_threshold = score_threshold
        self.lookahead_lines = lookahead_lines
        self.debug_match = debug_match
        self.use_line_number_annotator = use_line_number_annotator
        self.use_sub_annotations = use_sub_annotations
        self.line_number_annotator_config = line_number_annotator_config

    def get_match_detail_reporter(self):
        return get_match_detail_reporter(self.debug_match)

    def get_simple_annotator_config(
            self,
            xml_mapping: Dict[str, Dict[str, str]],
            **kwargs) -> SimpleSimpleMatchingConfig:
        return SimpleSimpleMatchingConfig(
            threshold=self.score_threshold,
            lookahead_sequence_count=self.lookahead_lines,
            tag_config_map=get_simple_tag_config_map(xml_mapping),
            use_sub_annotations=self.use_sub_annotations,
            **kwargs
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
        annotator_config: AnnotatorConfig) -> List[AbstractAnnotator]:

    annotators: List[AbstractAnnotator] = []
    if annotator_config.use_line_number_annotator:
        annotators.append(TextLineNumberAnnotator(
            config=annotator_config.line_number_annotator_config
        ))
    if xml_path:
        target_annotations = xml_root_to_target_annotations(
            parse_xml(xml_path).getroot(),
            xml_mapping
        )
        if annotator_config.matcher_name == MatcherNames.SIMPLE:
            annotators.append(SimpleMatchingAnnotator(
                target_annotations,
                config=annotator_config.get_simple_annotator_config(
                    xml_mapping=xml_mapping
                )
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


def _resolve_tag_expression_namespace(
        tag_expression: str,
        namespaces: Dict[str, str]) -> str:
    if not tag_expression or not namespaces:
        return tag_expression
    for ns_name, ns_url in namespaces.items():
        tag_expression = re.sub(
            r'\b' + re.escape(ns_name) + r':',
            '{' + ns_url + '}',
            tag_expression
        )
    return tag_expression


def _resolve_tag_to_tei_mapping_namespace(
        tag_to_tei_path_mapping: Dict[str, str],
        namespaces: Dict[str, str]) -> Dict[str, str]:
    return {
        key: _resolve_tag_expression_namespace(value, namespaces=namespaces)
        for key, value in tag_to_tei_path_mapping.items()
    }


class AbstractAnnotatePipelineFactory(ABC):
    def __init__(
            self,
            opt: argparse.Namespace,
            tei_filename_pattern: str,
            container_node_path: str,
            tag_to_tei_path_mapping: Dict[str, str] = None,
            output_fields: Optional[Set[str]] = None,
            preserve_sub_tags: bool = False,
            no_preserve_sub_fields: Set[str] = None,
            require_matching_fields: Set[str] = None,
            required_fields: Set[str] = None,
            namespaces: Dict[str, str] = None):
        self.tei_filename_pattern = tei_filename_pattern
        self.container_node_path = container_node_path
        self.tag_to_tei_path_mapping = _resolve_tag_to_tei_mapping_namespace(
            tag_to_tei_path_mapping,
            namespaces=namespaces
        )
        self.source_base_path = opt.source_base_path
        self.source_path = opt.source_path
        self.output_path = opt.output_path
        self.failed_output_path = opt.failed_output_path
        self.xml_path = opt.xml_path
        self.xml_filename_regex = opt.xml_filename_regex
        self.limit = opt.limit
        self.resume = opt.resume
        self.skip_errors = opt.skip_errors
        self.preserve_tags = not opt.no_preserve_tags
        self.always_preserve_fields = opt.always_preserve_fields
        self.preserve_sub_tags = preserve_sub_tags
        self.no_preserve_sub_fields = no_preserve_sub_fields
        self.require_matching_fields = require_matching_fields
        self.required_fields = required_fields
        self.output_fields = output_fields
        self.namespaces = namespaces
        self.annotator_config = AnnotatorConfig(
            matcher_name=opt.matcher,
            score_threshold=opt.matcher_score_threshold,
            lookahead_lines=opt.matcher_lookahead_lines,
            debug_match=opt.debug_match,
            use_line_number_annotator=opt.use_line_number_annotator,
            line_number_annotator_config=TextLineNumberAnnotatorConfig(
                min_line_number=opt.min_line_numbers_per_page,
                line_number_ratio_threshold=opt.min_line_number_ratio,
            )
        )
        self.file_exit_stack = ExitStack()

    @abstractmethod
    def get_annotator(self, source_url: str):
        pass

    def get_annotator_config(self) -> AnnotatorConfig:
        return self.annotator_config

    def get_tei_xml_output_file_for_source_file(self, source_url: str) -> str:
        return os.path.join(
            self.output_path,
            os.path.basename(source_url)
        )

    def get_tei_xml_failed_output_file_for_source_file(self, source_url: str) -> Optional[str]:
        if not self.failed_output_path:
            return None
        return os.path.join(
            self.failed_output_path,
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

    def is_structured_document_passing_checks(
            self,
            structured_document: GrobidTrainingTeiStructuredDocument,
            target_annotations: List[TargetAnnotation]) -> bool:
        return is_structured_document_passing_checks(
            structured_document,
            require_matching_fields=self.require_matching_fields,
            required_fields=self.required_fields,
            target_annotations=target_annotations
        )

    def get_final_source_url(self, source_url: str) -> str:
        return source_url

    def auto_annotate(self, source_url: str):
        with self.file_exit_stack:
            self._auto_annotate(source_url)

    def _auto_annotate(self, source_url: str):
        try:
            output_xml_path = self.get_tei_xml_output_file_for_source_file(source_url)
            final_source_url = self.get_final_source_url(source_url)
            annotator = self.get_annotator(final_source_url)
            annotate_structured_document(
                final_source_url,
                output_xml_path,
                annotator=annotator,
                preserve_tags=self.preserve_tags,
                fields=self.output_fields,
                always_preserve_fields=self.always_preserve_fields,
                container_node_path=self.container_node_path,
                tag_to_tei_path_mapping=self.tag_to_tei_path_mapping,
                preserve_sub_tags=self.preserve_sub_tags,
                no_preserve_sub_fields=self.no_preserve_sub_fields,
                is_structured_document_passing_checks=partial(
                    self.is_structured_document_passing_checks,
                    target_annotations=get_target_annotations_from_annotator(annotator)
                ),
                failed_target_structured_document_path=(
                    self.get_tei_xml_failed_output_file_for_source_file(source_url)
                ),
                namespaces=self.namespaces
            )
        except Exception as e:  # pylint: disable=broad-except
            skipping_msg = ' (skipping)' if self.skip_errors else ''
            get_logger().error(
                'failed to process %s%s due to %s',
                source_url, skipping_msg, e, exc_info=e
            )
            if not self.skip_errors:
                raise RuntimeError('failed to process %s due to %s' % (
                    source_url, e
                )) from e

    def get_source_file_list(self):
        if self.source_path:
            LOGGER.debug('using source_path: %r', self.source_path)
            return [self.source_path]
        LOGGER.debug(
            'finding files: source_base_path=%r, pattern=%r, limit=%s',
            self.source_base_path, self.tei_filename_pattern, self.limit
        )
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

    def configure_beam_pipeline(self, p: beam.Pipeline, tei_xml_file_list: List[str]):
        _ = (
            p
            | beam.Create(tei_xml_file_list)
            | "Auto-Annotate" >> beam.Map(self.auto_annotate)
        )

    def run_beam_pipeline(
            self,
            args: argparse.Namespace,
            tei_xml_file_list: List[str],
            save_main_session: bool = True):
        # We use the save_main_session option because one or more DoFn's in this
        # workflow rely on global context (e.g., a module imported at module level).
        pipeline_options = PipelineOptions.from_dictionary(vars(args))
        pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

        with beam.Pipeline(args.runner, options=pipeline_options) as p:
            self.configure_beam_pipeline(p, tei_xml_file_list=tei_xml_file_list)

            # Execute the pipeline and wait until it is completed.

    def run_local_pipeline(self, args: argparse.Namespace, tei_xml_file_list: List[str]):
        num_workers = min(args.num_workers, len(tei_xml_file_list))
        multi_processing = args.multi_processing
        LOGGER.info('using %d workers (multi_processing: %s)', num_workers, multi_processing)
        PoolExecutor = (
            concurrent.futures.ProcessPoolExecutor if multi_processing
            else concurrent.futures.ThreadPoolExecutor
        )
        with PoolExecutor(max_workers=num_workers) as executor:
            with logging_tqdm(total=len(tei_xml_file_list), logger=LOGGER) as pbar:
                future_to_url = {
                    executor.submit(self.auto_annotate, url): url
                    for url in tei_xml_file_list
                }
                LOGGER.debug('future_to_url: %s', future_to_url)
                for future in concurrent.futures.as_completed(future_to_url):
                    pbar.update(1)
                    future.result()

    def run(self, args: argparse.Namespace, save_main_session: bool = True):
        tei_xml_file_list = self.get_remaining_source_file_list()
        if not tei_xml_file_list:
            LOGGER.warning('no files to process')
            return

        if not args.cloud and args.num_workers >= 1:
            self.run_local_pipeline(args, tei_xml_file_list=tei_xml_file_list)
            return

        self.run_beam_pipeline(
            args,
            save_main_session=save_main_session,
            tei_xml_file_list=tei_xml_file_list
        )
