from __future__ import absolute_import

import argparse
import logging

from sciencebeam_gym.preprocess.annotation.annotator import Annotator

from .utils.string import comma_separated_str_to_list

from .structured_document.grobid_training_tei import (
    DEFAULT_TAG_KEY
)

from .utils.xml import parse_xml

from .annotation.target_annotation import (
    xml_root_to_target_annotations
)

from .annotation.simple_matching_annotator import (
    SimpleMatchingAnnotator
)

from .annotation.replace_tags_annotator import (
    ReplaceTagsAnnotatorConfig,
    ReplaceTagsPostProcessingAnnotator
)

from .annotation.expand_to_untagged_lines_annotator import (
    ExpandToUntaggedLinesAnnotatorConfig,
    ExpandToPreviousUntaggedLinesPostProcessingAnnotator,
    ExpandToFollowingUntaggedLinesPostProcessingAnnotator
)

from .auto_annotate_utils import (
    add_debug_argument,
    process_debug_argument,
    get_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    add_document_checks_arguments,
    process_annotation_pipeline_arguments,
    AnnotatorConfig,
    AbstractAnnotatePipelineFactory
)


LOGGER = logging.getLogger(__name__)


FULLTEXT_CONTAINER_NODE_PATH = 'text'


XREF_REL_TEI_PATH_MAPPING = {
    'xref-bib': 'ref[@type="biblio"]',
    'xref-figure': 'ref[@type="figure"]',
    'xref-table': 'ref[@type="table"]',
    'xref-formula': 'ref[@type="formula"]',
    'xref-section': 'ref[@type="section"]',
    'xref-box': 'ref[@type="box"]'
}


FULLTEXT_TAG_TO_TEI_PATH_MAPPING = {
    DEFAULT_TAG_KEY: 'other',
    'note_other': 'note[@type="other"]',
    'section_title': 'head',
    'section_paragraph': 'p',
    **{
        'section_paragraph-%s' % key: 'p/%s' % value
        for key, value in XREF_REL_TEI_PATH_MAPPING.items()
    },
    'figure': 'figure',
    'table': 'figure[@type="table"]',
    # Note: we are not using `<figure type="box">` because that is not supported yet
    'boxed_text_title': 'head[@type="box"]',
    'boxed_text_paragraph': 'p[@type="box"]',
    **{
        'boxed_text_paragraph-%s' % key: 'p[@type="box"]/%s' % value
        for key, value in XREF_REL_TEI_PATH_MAPPING.items()
    },
}


REPLACED_TAG_BY_TAG_MAP = {
    'note_other': None,
    'note[@type="other"]': None,
}


# Where the reference XML might just contain an image, we are assuming that all
# untagged lines after the label and caption also belong to the same element
EXPAND_TO_UNTAGGED_LINES_ENABLED_TAGS = {
    'figure',
    'table'
}


def _get_annotator(
        xml_path,
        xml_mapping,
        annotator_config: AnnotatorConfig):
    target_annotations = xml_root_to_target_annotations(
        parse_xml(xml_path).getroot(),
        xml_mapping
    )
    simple_annotator_config = annotator_config.get_simple_annotator_config(
        xml_mapping=xml_mapping,
        preserve_sub_annotations=True,
        extend_to_line_enabled=False
    )
    annotators = [
        SimpleMatchingAnnotator(
            target_annotations,
            config=simple_annotator_config
        ),
        ReplaceTagsPostProcessingAnnotator(
            config=ReplaceTagsAnnotatorConfig(
                replaced_tag_by_tag=REPLACED_TAG_BY_TAG_MAP
            )
        ),
        ExpandToPreviousUntaggedLinesPostProcessingAnnotator(
            config=ExpandToUntaggedLinesAnnotatorConfig(
                enabled_tags=EXPAND_TO_UNTAGGED_LINES_ENABLED_TAGS
            )
        ),
        ExpandToFollowingUntaggedLinesPostProcessingAnnotator(
            config=ExpandToUntaggedLinesAnnotatorConfig(
                enabled_tags=EXPAND_TO_UNTAGGED_LINES_ENABLED_TAGS
            )
        )
    ]
    annotator = Annotator(annotators)
    return annotator


class AnnotatePipelineFactory(AbstractAnnotatePipelineFactory):
    def __init__(self, opt):
        super().__init__(
            opt,
            tei_filename_pattern='*.fulltext.tei.xml*',
            container_node_path=FULLTEXT_CONTAINER_NODE_PATH,
            tag_to_tei_path_mapping=FULLTEXT_TAG_TO_TEI_PATH_MAPPING,
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
        self.annotator_config.use_sub_annotations = True

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

    parser.add_argument(
        '--fields',
        type=comma_separated_str_to_list,
        default=','.join([
            'section_title',
            'section_paragraph',
            'boxed_text_title',
            'boxed_text_paragraph',
            'figure',
            'table'
        ]),
        help='comma separated list of fields to annotate'
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
