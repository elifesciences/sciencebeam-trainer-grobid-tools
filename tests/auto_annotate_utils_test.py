import argparse

import pytest

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    get_filtered_xml_mapping_and_fields,
    add_annotation_pipeline_arguments,
    process_annotation_pipeline_arguments
)

from .test_utils import dict_to_args


DEFAULT_ARGS_DICT = {
    'source-base-path': 'source-base-path1',
    'output-path': 'output-path1',
    'xml-path': 'xml-path1',
    'xml-filename-regex': 'xml-filename-regex1'
}


def _parse_annotation_pipeline_arguments(argv):
    parser = argparse.ArgumentParser()
    add_annotation_pipeline_arguments(parser)
    args = parser.parse_args(argv)
    process_annotation_pipeline_arguments(parser, args)
    return args


class TestAnnotationPipelineArguments:
    def test_should_parse_required_args(self):
        args = _parse_annotation_pipeline_arguments(dict_to_args(DEFAULT_ARGS_DICT))
        assert args.source_base_path == DEFAULT_ARGS_DICT['source-base-path']

    def test_should_require_source(self):
        with pytest.raises(SystemExit):
            _parse_annotation_pipeline_arguments(dict_to_args({
                **DEFAULT_ARGS_DICT,
                'source-base-path': None
            }))


class TestGetFilteredXmlMappingAndFields:
    def test_should_filter_rops(self):
        xml_mapping = {
            'any': {
                'tag1': 'xpath1',
                'tag2': 'xpath2'
            }
        }
        expected_xml_mapping = {
            'any': {
                'tag1': 'xpath1'
            }
        }
        fields = ['tag1']
        assert get_filtered_xml_mapping_and_fields(
            xml_mapping, fields
        ) == (expected_xml_mapping, fields)

    def test_should_include_related_props(self):
        xml_mapping = {
            'any': {
                'tag1': 'xpath1',
                'tag1.related': 'related1'
            }
        }
        fields = ['tag1']
        assert get_filtered_xml_mapping_and_fields(
            xml_mapping, fields
        ) == (xml_mapping, fields)

    def test_should_include_keys_with_dot_as_fields(self):
        xml_mapping = {
            'any': {
                'tag1': 'xpath1',
                'tag1.related': 'related1'
            }
        }
        assert get_filtered_xml_mapping_and_fields(
            xml_mapping, None
        ) == (xml_mapping, {'tag1'})
