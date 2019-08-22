from __future__ import absolute_import

import argparse
import logging

from sciencebeam_gym.preprocess.annotation.target_annotation import (
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

from .structured_document.grobid_training_tei import ContainerNodePaths
from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    load_xml,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments,
    AbstractAnnotatePipelineFactory
)


LOGGER = logging.getLogger(__name__)


HEADER_CONTAINER_NODE_PATH = ContainerNodePaths.HEADER_CONTAINER_NODE_PATH


def get_logger():
    return logging.getLogger(__name__)


def _get_annotator(
        xml_path, xml_mapping, match_detail_reporter,
        use_tag_begin_prefix=False,
        use_line_no_annotator=False):

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
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.header.tei.xml*',
            container_node_path=HEADER_CONTAINER_NODE_PATH,
            output_fields=opt.fields
        )
        self.xml_mapping, self.fields = get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields
        )

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            match_detail_reporter=None
        )


def add_main_args(parser):
    add_annotation_pipeline_arguments(parser)

    parser.add_argument(
        '--fields',
        type=comma_separated_str_to_list,
        help='comma separated list of fields to annotate'
    )

    add_debug_argument(parser)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)

    parsed_args = parser.parse_args(argv)
    process_annotation_pipeline_arguments(parsed_args)
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
