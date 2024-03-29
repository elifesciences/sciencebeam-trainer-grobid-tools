from __future__ import absolute_import

import argparse
import logging

from .core.annotation.annotator import Annotator

from .structured_document.grobid_training_tei import (
    ContainerNodePaths,
    DEFAULT_TAG_KEY
)

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    add_document_checks_arguments,
    add_fields_argument,
    process_annotation_pipeline_arguments,
    get_default_annotators,
    AbstractAnnotatePipelineFactory
)


LOGGER = logging.getLogger(__name__)


HEADER_CONTAINER_NODE_PATH = ContainerNodePaths.HEADER_CONTAINER_NODE_PATH


HEADER_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'note[@type="other"]',
    'title': 'docTitle/titlePart',
    'abstract': 'div[@type="abstract"]',
    'author': 'byline/docAuthor',
    'author_aff': 'byline/affiliation',
    'line_no': 'note[@type="line_no"]'
}


def get_logger():
    return logging.getLogger(__name__)


def _get_annotator(*args, **kwargs):
    annotators = get_default_annotators(*args, **kwargs)
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.header.tei.xml*',
            container_node_path=HEADER_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=HEADER_TAG_TO_TEI_PATH_MAPPING,
            require_matching_fields=opt.require_matching_fields,
            required_fields=opt.required_fields,
            output_fields=opt.fields
        )
        self.xml_mapping, self.fields = get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields,
            xml_mapping_overrides=opt.xml_mapping_overrides
        )
        self.tag_to_tei_path_mapping = self.tag_to_tei_path_mapping.copy()
        for field in self.fields:
            if field not in self.tag_to_tei_path_mapping:
                self.tag_to_tei_path_mapping[field] = 'note[@type="%s"]' % field

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config()
        )


def add_main_args(parser):
    add_annotation_pipeline_arguments(parser)
    add_document_checks_arguments(parser)
    add_fields_argument(parser)
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
