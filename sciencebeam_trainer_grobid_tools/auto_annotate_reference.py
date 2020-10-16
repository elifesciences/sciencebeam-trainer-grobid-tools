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

from .structured_document.reference_annotator import (
    DEFAULT_IDNO_PREFIX_REGEX,
    ReferenceAnnotatorConfig,
    ReferenceSubTagOnlyAnnotator,
    ReferencePostProcessingAnnotator
)


LOGGER = logging.getLogger(__name__)


REFERENCE_CONTAINER_NODE_PATH = 'tei:text/tei:back/tei:listBibl'


REFERENCE_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'tei:note[@type="other"]',
    'reference': 'tei:bibl',
    'reference-label': 'tei:bibl/tei:label',
    'reference-author': 'tei:bibl/tei:author',
    'reference-editor': 'tei:bibl/tei:editor',
    'reference-year': 'tei:bibl/tei:date',
    'reference-article-title': 'tei:bibl/tei:title[@level="a"]',
    'reference-source': 'tei:bibl/tei:title[@level="j"]',
    'reference-publisher-name': 'tei:bibl/tei:publisher',
    'reference-publisher-loc': 'tei:bibl/tei:pubPlace',
    'reference-volume': 'tei:bibl/tei:biblScope[@unit="volume"]',
    'reference-issue': 'tei:bibl/tei:biblScope[@unit="issue"]',
    'reference-page': 'tei:bibl/tei:biblScope[@unit="page"]',
    'reference-issn': 'tei:bibl/tei:idno[@type="ISSN"]',
    'reference-isbn': 'tei:bibl/tei:idno[@type="ISBN"]',
    'reference-doi': 'tei:bibl/tei:idno[@type="DOI"]',
    'reference-pii': 'tei:bibl/tei:idno[@type="PII"]',
    'reference-pmid': 'tei:bibl/tei:idno[@type="PMID"]',
    'reference-pmcid': 'tei:bibl/tei:idno[@type="PMC"]',
    'reference-arxiv': 'tei:bibl/tei:idno[@type="arxiv"]',
    'ext-link': 'tei:bibl/tei:ptr[@type="web"]'
}

DEFAULT_SUB_TAG_MAP = {
    'reference-fpage': 'reference-page',
    'reference-lpage': 'reference-page'
}

DEFAULT_MERGE_ENABLED_SUB_TAGS = {
    'reference-author',
    'reference-editor',
    'reference-issue',
    'reference-page'
}

NAME_SUFFIX_ENABLED_SUB_TAGS = {
    'reference-author',
    'reference-editor'
}

IDNO_SUB_TAGS = {
    'reference-issn',
    'reference-isbn',
    'reference-doi',
    'reference-pii',
    'reference-pmid',
    'reference-pmcid',
    'reference-arxiv'
}

IDNO_PREFIX_REGEX_MAP = {
    'reference-issn': DEFAULT_IDNO_PREFIX_REGEX,
    'reference-isbn': DEFAULT_IDNO_PREFIX_REGEX,
    'reference-doi': r'(?i)\bDOI(\s?:)?$',
    'reference-pii': r'(?i)\bPII(\s?:)?$',
    'reference-pmid': DEFAULT_IDNO_PREFIX_REGEX,
    'reference-pmcid': DEFAULT_IDNO_PREFIX_REGEX,
    'reference-arxiv': DEFAULT_IDNO_PREFIX_REGEX
}

ETAL_SUB_TAG = 'reference-etal'

ETAL_MERGE_ENABLED_SUB_TAGS = {
    'reference-author',
    'reference-editor'
}


def get_logger():
    return logging.getLogger(__name__)


def _get_default_reference_annotator_config() -> ReferenceAnnotatorConfig:
    return ReferenceAnnotatorConfig(
        sub_tag_map=DEFAULT_SUB_TAG_MAP,
        merge_enabled_sub_tags=DEFAULT_MERGE_ENABLED_SUB_TAGS,
        include_prefix_enabled_sub_tags={},
        include_suffix_enabled_sub_tags=NAME_SUFFIX_ENABLED_SUB_TAGS,
        prefix_regex_by_sub_tag_map=IDNO_PREFIX_REGEX_MAP,
        etal_sub_tag=ETAL_SUB_TAG,
        etal_merge_enabled_sub_tags=ETAL_MERGE_ENABLED_SUB_TAGS,
        remove_untagged_enabled=False
    )


def _get_annotator(
        xml_path,
        xml_mapping,
        annotator_config: AnnotatorConfig,
        reference_annotator_config: ReferenceAnnotatorConfig,
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
        annotators.append(ReferenceSubTagOnlyAnnotator(
            target_annotations,
            config=simple_annotator_config
        ))
    annotators.append(
        ReferencePostProcessingAnnotator(
            reference_annotator_config
        )
    )
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
            tei_filename_pattern='*.references.tei.xml*',
            container_node_path=REFERENCE_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=_resolve_tag_to_tei_mapping_namespace(
                REFERENCE_TAG_TO_TEI_PATH_MAPPING
            ),
            output_fields=opt.fields,
            namespaces=TEI_NS_MAP
        )
        self.segment_references = opt.segment_references
        if not opt.segment_references:
            self.always_preserve_fields = ['reference']
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
        self.reference_annotator_config = _get_default_reference_annotator_config()
        if opt.include_idno_prefix:
            self.reference_annotator_config.include_prefix_enabled_sub_tags = IDNO_SUB_TAGS
        self.remove_untagged_enabled = opt.remove_invalid_references

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config(),
            reference_annotator_config=self.reference_annotator_config,
            segment_references=self.segment_references,
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
        '--include-idno-prefix',
        action='store_true',
        default=False,
        help='enable including the prefix of an idno, e.g. "doi:"'
    )

    parser.add_argument(
        '--segment-references',
        action='store_true',
        default=False,
        help='enable segmentation of references. bibl element will be set or replaced by note.'
    )

    parser.add_argument(
        '--remove-invalid-references',
        action='store_true',
        default=False,
        help=(
            'enable removing invalid references'
            + ' (usually in combination with --segment-references).'
        )
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
