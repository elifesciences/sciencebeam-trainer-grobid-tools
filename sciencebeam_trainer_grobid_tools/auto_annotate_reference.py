from __future__ import absolute_import

import argparse
import logging
import re
from typing import Dict

from sciencebeam_gym.preprocess.annotation.annotator import Annotator

from .utils.string import comma_separated_str_to_list

from .structured_document.grobid_training_tei import (
    DEFAULT_TAG_KEY
)

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments,
    get_default_annotators,
    AbstractAnnotatePipelineFactory
)

from .structured_document.reference_annotator import (
    ReferenceAnnotatorConfig,
    ReferencePostProcessingAnnotator
)


LOGGER = logging.getLogger(__name__)


TEI_NS = 'http://www.tei-c.org/ns/1.0'

TEI_NS_MAP = {
    'tei': TEI_NS
}


REFERENCE_CONTAINER_NODE_PATH = 'tei:text/tei:back'


REFERENCE_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'tei:note[@type="other"]',
    'reference': 'tei:listBibl/tei:bibl',
    'reference-label': 'tei:listBibl/tei:bibl/tei:label',
    'reference-author': 'tei:listBibl/tei:bibl/tei:author',
    'reference-editor': 'tei:listBibl/tei:bibl/tei:editor',
    'reference-year': 'tei:listBibl/tei:bibl/tei:date',
    'reference-article-title': 'tei:listBibl/tei:bibl/tei:title[@level="a"]',
    'reference-source': 'tei:listBibl/tei:bibl/tei:title[@level="j"]',
    'reference-publisher-name': 'tei:listBibl/tei:bibl/tei:publisher',
    'reference-publisher-loc': 'tei:listBibl/tei:bibl/tei:pubPlace',
    'reference-volume': 'tei:listBibl/tei:bibl/tei:biblScope[@unit="volume"]',
    'reference-issue': 'tei:listBibl/tei:bibl/tei:biblScope[@unit="issue"]',
    'reference-page': 'tei:listBibl/tei:bibl/tei:biblScope[@unit="page"]',
    'reference-issn': 'tei:listBibl/tei:bibl/tei:idno[@type="ISSN"]',
    'reference-isbn': 'tei:listBibl/tei:bibl/tei:idno[@type="ISBN"]',
    'reference-doi': 'tei:listBibl/tei:bibl/tei:idno[@type="DOI"]',
    'reference-pmid': 'tei:listBibl/tei:bibl/tei:idno[@type="PMID"]',
    'reference-pmcid': 'tei:listBibl/tei:bibl/tei:idno[@type="PMC"]',
    'reference-arxiv': 'tei:listBibl/tei:bibl/tei:idno[@type="arxiv"]',
    'ext-link': 'tei:listBibl/tei:bibl/tei:ptr[@type="web"]'
}

DEFAULT_SUB_TAG_MAP = {
    'reference-fpage': 'reference-page',
    'reference-lpage': 'reference-page'
}

DEFAULT_MERGE_ENABLED_SUB_TAGS = {
    'reference-author',
    'reference-editor',
    'reference-page'
}

IDNO_SUB_TAGS = {
    'reference-issn',
    'reference-isbn',
    'reference-doi',
    'reference-pmid',
    'reference-pmcid',
    'reference-arxiv'
}


def get_logger():
    return logging.getLogger(__name__)


def _get_default_reference_annotator_config() -> ReferenceAnnotatorConfig:
    return ReferenceAnnotatorConfig(
        sub_tag_map=DEFAULT_SUB_TAG_MAP,
        merge_enabled_sub_tags=DEFAULT_MERGE_ENABLED_SUB_TAGS,
        include_prefix_enabled_sub_tags={}
    )


def _get_annotator(
        *args,
        reference_annotator_config: ReferenceAnnotatorConfig,
        **kwargs):
    annotators = get_default_annotators(*args, **kwargs)
    annotators = annotators + [
        ReferencePostProcessingAnnotator(
            reference_annotator_config
        )
    ]
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

    def get_annotator(self, source_url: str):
        target_xml_path = self.get_target_xml_for_source_file(source_url)
        return _get_annotator(
            target_xml_path,
            self.xml_mapping,
            annotator_config=self.get_annotator_config(),
            reference_annotator_config=self.reference_annotator_config
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
