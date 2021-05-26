from __future__ import absolute_import

import argparse
import logging
from typing import List, Optional

from .core.annotation.annotator import AbstractAnnotator, Annotator

from .utils.xml import parse_xml
from .utils.tei_xml import TEI_NS_MAP

from .structured_document.grobid_training_tei import (
    DEFAULT_TAG_KEY,
    SUB_LEVEL
)

from .annotation.target_annotation import (
    xml_root_to_target_annotations
)

from .annotation.sub_tag_annotator import SubTagOnlyAnnotator
from .annotation.remove_untagged_annotator import RemoveUntaggedPostProcessingAnnotator

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

from .annotation.merge_group_tags_annotator import (
    MergeGroupTagsAnnotatorConfig,
    MergeGroupTagsPostProcessingAnnotator
)


LOGGER = logging.getLogger(__name__)


AFFILIATION_CONTAINER_NODE_PATH = (
    'tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author'
)


AFFILIATION_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'tei:note[@type="other"]',
    'author_aff': 'tei:affiliation',
    'author_aff-label': 'tei:affiliation/tei:marker',
    'author_aff-department': 'tei:affiliation/tei:orgName[@type="department"]',
    'author_aff-institution': 'tei:affiliation/tei:orgName[@type="institution"]',
    'author_aff-address': 'tei:affiliation/tei:address',
    'author_aff-address-city': 'tei:affiliation/tei:address/tei:settlement',
    'author_aff-address-postcode': 'tei:affiliation/tei:address/tei:postCode',
    'author_aff-address-state': 'tei:affiliation/tei:address/tei:region',
    'author_aff-address-country': 'tei:affiliation/tei:address/tei:country'
}


DEFAULT_AFFILIATION_FIELDS = ['author_aff']


def is_address_sub_tag(sub_tag: str) -> bool:
    # this includes the defined address fields as well as unknown preserved sub tags
    # that will have the full namespace
    return 'address' in sub_tag


def get_address_group_tag_for_sub_tag(sub_tag: str) -> Optional[str]:
    if is_address_sub_tag(sub_tag):
        return 'author_aff-address'
    return None


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
        preserve_sub_annotations=True,
        extend_to_line_enabled=False
    )
    annotators: List[AbstractAnnotator] = []
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
    annotators.append(MergeGroupTagsPostProcessingAnnotator(
        config=MergeGroupTagsAnnotatorConfig(
            get_group_tag_for_tag_fn=get_address_group_tag_for_sub_tag,
            tag_level=SUB_LEVEL
        )
    ))
    # annotators.append(AffiliationAddressPostProcessingAnnotator(AffiliationAddressAnnotatorConfig(
    #     address_sub_tag='author_aff-address',
    #     is_address_sub_tag_fn=is_address_sub_tag
    # )))
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.affiliation.tei.xml*',
            container_node_path=AFFILIATION_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=AFFILIATION_TAG_TO_TEI_PATH_MAPPING,
            output_fields=opt.fields,
            preserve_sub_tags=opt.preserve_sub_tags,
            no_preserve_sub_fields=opt.no_preserve_sub_fields,
            namespaces=TEI_NS_MAP
        )
        self.segment_affiliation = opt.segment_affiliation
        if not opt.segment_affiliation:
            self.always_preserve_fields = ['author_aff']
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
    add_fields_argument(parser, default_fields=DEFAULT_AFFILIATION_FIELDS)
    add_sub_fields_argument(parser)
    add_preserve_sub_tags_argument(parser)
    add_no_preserve_sub_fields_argument(parser)

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
