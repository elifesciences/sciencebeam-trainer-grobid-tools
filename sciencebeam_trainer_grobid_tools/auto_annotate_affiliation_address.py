from __future__ import absolute_import

import argparse
import logging
import re
from typing import Dict

from sciencebeam_gym.preprocess.annotation.annotator import Annotator

from .utils.string import comma_separated_str_to_list
from .utils.xml import parse_xml
from .utils.tei_xml import TEI_NS, TEI_NS_MAP

from .structured_document.grobid_training_tei import (
    DEFAULT_TAG_KEY
)

from .annotation.target_annotation import (
    xml_root_to_target_annotations
)

from .annotation.sub_tag_annotator import SubTagOnlyAnnotator
from .annotation.remove_untagged_annotator import RemoveUntaggedPostProcessingAnnotator

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments,
    AbstractAnnotatePipelineFactory,
    AnnotatorConfig
)

from .structured_document.simple_matching_annotator import (
    SimpleMatchingAnnotator
)


LOGGER = logging.getLogger(__name__)


AFFILIATION_CONTAINER_NODE_PATH = (
    'tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author'
)


AFFILIATION_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'tei:note[@type="other"]',
    'author_aff': 'tei:affiliation',
    'author_aff-label': 'tei:affiliation/tei:marker',
}


def _get_annotator(
        xml_path,
        xml_mapping,
        annotator_config: AnnotatorConfig,
        segment_references: bool,
        remove_untagged_enabled: bool):
    target_annotations = xml_root_to_target_annotations(
        parse_xml(xml_path).getroot(),
        xml_mapping
    )
    simple_annotator_config = annotator_config.get_simple_annotator_config(
        xml_mapping=xml_mapping,
        extend_to_line_enabled=False
    )
    annotators = []
    if segment_references:
        annotators.append(SimpleMatchingAnnotator(
            target_annotations,
            config=simple_annotator_config
        ))
    else:
        annotators.append(SubTagOnlyAnnotator(
            target_annotations,
            config=simple_annotator_config
        ))
    if remove_untagged_enabled:
        annotators.append(RemoveUntaggedPostProcessingAnnotator())
    annotator = Annotator(annotators)
    return annotator


def _resolve_tag_expression_namespace(
        tag_expression: str) -> str:
    if not tag_expression:
        return tag_expression
    return re.sub(
        r'\btei:',
        '{' + TEI_NS + '}',
        tag_expression
    )


def _resolve_tag_to_tei_mapping_namespace(
        tag_to_tei_path_mapping: Dict[str, str]) -> Dict[str, str]:
    return {
        key: _resolve_tag_expression_namespace(value)
        for key, value in tag_to_tei_path_mapping.items()
    }


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.affiliation.tei.xml*',
            container_node_path=AFFILIATION_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=_resolve_tag_to_tei_mapping_namespace(
                AFFILIATION_TAG_TO_TEI_PATH_MAPPING
            ),
            output_fields=opt.fields,
            namespaces=TEI_NS_MAP
        )
        self.segment_affiliation = opt.segment_affiliation
        if not opt.segment_affiliation:
            self.always_preserve_fields = ['author_aff']
        self.xml_mapping, self.fields = get_xml_mapping_and_fields(
            opt.xml_mapping_path,
            opt.fields,
            xml_mapping_overrides=opt.xml_mapping_overrides
        )
        self.tag_to_tei_path_mapping = self.tag_to_tei_path_mapping.copy()
        for field in self.fields:
            if field not in self.tag_to_tei_path_mapping:
                self.tag_to_tei_path_mapping[field] = 'note[type="%s"]' % field
        self.annotator_config.use_sub_annotations = True
        self.remove_untagged_enabled = opt.remove_invalid_affiliations

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config(),
            segment_references=self.segment_affiliation,
            remove_untagged_enabled=self.remove_untagged_enabled
        )


def add_main_args(parser):
    add_annotation_pipeline_arguments(parser)

    parser.add_argument(
        '--fields',
        type=comma_separated_str_to_list,
        default='reference',
        help='comma separated list of fields to annotate'
    )

    parser.add_argument(
        '--segment-affiliation',
        action='store_true',
        default=False,
        help=(
            'enable segmentation of affiliations.'
            ' affiliation element will be set or replaced by note.'
        )
    )

    parser.add_argument(
        '--remove-invalid-affiliations',
        action='store_true',
        default=False,
        help=(
            'enable removing invalid affiliations'
            + ' (usually in combination with --segment-affiliation).'
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
