from __future__ import absolute_import

import argparse
import logging

from sciencebeam_gym.preprocess.annotation.annotator import Annotator

from .utils.xml import parse_xml
from .utils.tei_xml import TEI_NS_MAP

from .structured_document.grobid_training_tei import (
    DEFAULT_TAG_KEY
)

from .annotation.target_annotation import (
    xml_root_to_target_annotations
)

from .annotation.sub_tag_annotator import SubTagOnlyAnnotator

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    add_fields_argument,
    add_sub_fields_argument,
    add_preserve_sub_tags_argument,
    add_no_preserve_sub_fields_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments,
    AbstractAnnotatePipelineFactory,
    AnnotatorConfig
)

from .annotation.simple_matching_annotator import (
    SimpleMatchingAnnotator
)


LOGGER = logging.getLogger(__name__)


TABLE_CONTAINER_NODE_PATH = 'text'


TABLE_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'note[@type="other"]',
    'table': 'figure[@type="table"]',
    'table-label': 'figure[@type="table"]/head/label',
    'table-caption': 'figure[@type="table"]/figDesc',
}


DEFAULT_TABLE_FIELDS = ['table']


def _get_annotator(
        xml_path,
        xml_mapping,
        annotator_config: AnnotatorConfig,
        segment_tables: bool):
    target_annotations = xml_root_to_target_annotations(
        parse_xml(xml_path).getroot(),
        xml_mapping
    )
    simple_annotator_config = annotator_config.get_simple_annotator_config(
        xml_mapping=xml_mapping,
        preserve_sub_annotations=True,
        extend_to_line_enabled=False
    )
    annotators = []
    if segment_tables:
        annotators.append(SimpleMatchingAnnotator(
            target_annotations,
            config=simple_annotator_config
        ))
    else:
        annotators.append(SubTagOnlyAnnotator(
            target_annotations,
            config=simple_annotator_config
        ))
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.table.tei.xml*',
            container_node_path=TABLE_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=TABLE_TAG_TO_TEI_PATH_MAPPING,
            output_fields=opt.fields,
            preserve_sub_tags=opt.preserve_sub_tags,
            no_preserve_sub_fields=opt.no_preserve_sub_fields,
            namespaces=TEI_NS_MAP
        )
        self.segment_tables = opt.segment_tables
        if not opt.segment_tables:
            self.always_preserve_fields = ['table']
        self.xml_mapping, self.fields = get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields,
            sub_fields=opt.sub_fields,
            xml_mapping_overrides=opt.xml_mapping_overrides
        )
        self.tag_to_tei_path_mapping = self.tag_to_tei_path_mapping.copy()
        for field in self.fields:
            if field not in self.tag_to_tei_path_mapping:
                self.tag_to_tei_path_mapping[field] = 'note[@type="%s"]' % field
        self.annotator_config.use_sub_annotations = True

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config(),
            segment_tables=self.segment_tables
        )


def add_main_args(parser):
    add_annotation_pipeline_arguments(parser)
    add_fields_argument(parser, default_fields=DEFAULT_TABLE_FIELDS)
    add_sub_fields_argument(parser)
    add_preserve_sub_tags_argument(parser)
    add_no_preserve_sub_fields_argument(parser)

    parser.add_argument(
        '--segment-tables',
        action='store_true',
        default=False,
        help=(
            'enable segmentation of tables.'
        )
    )

    add_debug_argument(parser)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)

    parsed_args = parser.parse_args(argv)
    process_annotation_pipeline_arguments(parser, parsed_args)
    LOGGER.info('parsed_args: %s', parsed_args)
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
