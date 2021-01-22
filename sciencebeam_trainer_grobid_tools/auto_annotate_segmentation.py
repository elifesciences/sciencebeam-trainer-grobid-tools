from __future__ import absolute_import

import argparse
import logging
from typing import List, Set

from sciencebeam_gym.preprocess.annotation.annotator import Annotator

from .utils.string import comma_separated_str_to_list

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments,
    add_document_checks_arguments,
    add_fields_argument,
    get_default_annotators,
    get_default_config_path,
    AbstractAnnotatePipelineFactory
)
from .structured_document.grobid_training_tei import (
    ContainerNodePaths,
    DEFAULT_TAG_KEY
)
from .annotation.segmentation_annotator import (
    SegmentationAnnotator,
    SegmentationConfig,
    parse_segmentation_config
)
from .annotation.target_annotation import TargetAnnotation
from .annotation.checks import (
    get_required_target_value_by_name,
    get_structured_document_entities_by_name
)

from .structured_document.grobid_training_tei import GrobidTrainingTeiStructuredDocument
from .utils.fuzzy import fuzzy_search


LOGGER = logging.getLogger(__name__)


SEGMENTATION_CONTAINER_NODE_PATH = ContainerNodePaths.SEGMENTATION_CONTAINER_NODE_PATH


SEGMENTATION_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'body',
    'acknowledgment': 'div[@type="acknowledgment"]',
    'annex': 'div[@type="annex"]',
    'line_no': 'note[@type="line_no"]',
    'reference': 'listBibl'
}


DEFAULT_SEGMENTATION_CONFIG = 'segmentation.conf'


DEFAULT_FIELDS = [
    'title',
    'abstract',
    'keywords_title',
    'keywords',
    'manuscript_type',
    'author',
    'author_aff',
    'author_notes',
    'body_section_title',
    'body_section_paragraph',
    'figure',
    'table',
    'back_section_title',
    'back_section_paragraph',
    'acknowledgment_section_title',
    'acknowledgment_section_paragraph',
    'appendix_group_title',
    'appendix',
    'reference'
]


def get_logger():
    return logging.getLogger(__name__)


def _get_annotator(
        *args,
        segmentation_config: SegmentationConfig = None,
        preserve_tags: bool = False,
        **kwargs):

    annotators = get_default_annotators(*args, **kwargs)
    annotators = annotators + [
        SegmentationAnnotator(segmentation_config, preserve_tags=preserve_tags)
    ]
    annotator = Annotator(annotators)
    return annotator


def is_segmentation_structured_document_passing_checks(
        structured_document: GrobidTrainingTeiStructuredDocument,
        require_matching_fields: Set[str],
        required_fields: Set[str],
        target_annotations: List[TargetAnnotation],
        segmentation_config: SegmentationConfig,
        threshold: float = 0.8) -> bool:
    """
    Segmentation checks are slightly different, because fields like
    "title" and "abstract" are all put into "front".
    Therefore we need to map those fields and do a fuzzy search.
    """
    require_matching_fields = set(require_matching_fields or set()) | set(required_fields or set())
    if not require_matching_fields:
        return True
    if not target_annotations:
        raise RuntimeError('target_annotations required')
    required_value_by_name = get_required_target_value_by_name(
        target_annotations=target_annotations,
        require_matching_fields=require_matching_fields
    )
    LOGGER.debug('required_fields: %s', required_fields)
    if required_fields:
        missing_required_fields = set(required_fields) - set(required_value_by_name.keys())
        if missing_required_fields:
            LOGGER.warning('missing_required_fields: %s', missing_required_fields)
            return False
    if not required_value_by_name:
        return True
    entities_by_name = get_structured_document_entities_by_name(structured_document)
    LOGGER.info('entities_by_name: %s', entities_by_name)
    expected_entity_by_field_name = {
        field_name: entity_name
        for entity_name, field_names in segmentation_config.segmentation_mapping.items()
        for field_name in field_names
    }
    for require_matching_field, required_value in required_value_by_name.items():
        expected_entity_name = expected_entity_by_field_name[require_matching_field]
        actual_entity_values = entities_by_name.get(expected_entity_name, [])
        if not actual_entity_values:
            LOGGER.warning(
                'required field not in tagged entities: %s -> %s',
                require_matching_field, expected_entity_name
            )
            return False
        actual_entity_joined_values = ' '.join(actual_entity_values)
        match_result = fuzzy_search(
            actual_entity_joined_values, required_value,
            threshold=threshold
        )
        if not match_result:
            LOGGER.warning(
                'required field found, but not matching (%s): %r !~ %r',
                require_matching_field, required_value, actual_entity_joined_values
            )
            return False
    return True


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.segmentation.tei.xml*',
            container_node_path=SEGMENTATION_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=SEGMENTATION_TAG_TO_TEI_PATH_MAPPING,
            require_matching_fields=opt.require_matching_fields,
            required_fields=opt.required_fields,
            output_fields=opt.no_preserve_fields
        )
        self.xml_mapping, self.fields = get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields,
            xml_mapping_overrides=opt.xml_mapping_overrides
        )
        self.segmentation_config = parse_segmentation_config(opt.segmentation_config)

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config(),
            segmentation_config=self.segmentation_config,
            preserve_tags=self.preserve_tags
        )

    def is_structured_document_passing_checks(
            self,
            structured_document: GrobidTrainingTeiStructuredDocument,
            target_annotations: List[TargetAnnotation]) -> bool:
        return is_segmentation_structured_document_passing_checks(
            structured_document,
            require_matching_fields=self.require_matching_fields,
            required_fields=self.required_fields,
            segmentation_config=self.segmentation_config,
            target_annotations=target_annotations
        )


def add_main_args(parser):
    add_annotation_pipeline_arguments(parser)
    add_document_checks_arguments(parser)
    add_fields_argument(parser, default_fields=DEFAULT_FIELDS)

    parser.add_argument(
        '--no-preserve-fields',
        type=comma_separated_str_to_list,
        help='comma separated list of output fields that should not be preserved'
    )

    parser.add_argument(
        '--segmentation-config',
        default=get_default_config_path(DEFAULT_SEGMENTATION_CONFIG),
        help='path to segmentation config'
    )

    add_debug_argument(parser)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)

    parsed_args = parser.parse_args(argv)
    process_annotation_pipeline_arguments(parser, parsed_args)
    get_logger().info('parsed_args: %s', parsed_args)
    return parsed_args


def run(args: argparse.Namespace, save_main_session: bool = True):
    AnnotatePipelineFactory(args).run(args, save_main_session=save_main_session)


def main(argv=None, save_main_session: bool = True):
    args = parse_args(argv)
    process_debug_argument(args)
    run(args, save_main_session=save_main_session)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
