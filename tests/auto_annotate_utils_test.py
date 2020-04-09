import argparse

import pytest

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    get_filtered_xml_mapping_and_fields,
    get_xml_mapping_with_overrides,
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

    def test_should_parse_multiple_xml_mapping_overrides(self):
        argv = list(dict_to_args(DEFAULT_ARGS_DICT))
        argv.extend(['--xml-mapping-overrides', 'key1=value1|key2=value2'])
        args = _parse_annotation_pipeline_arguments(argv)
        assert args.xml_mapping_overrides == {
            'key1': 'value1',
            'key2': 'value2'
        }


class TestGetFilteredXmlMappingAndFields:
    def test_should_filter_props(self):
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


class TestGetXmlMappingWithOverrides:
    def test_should_return_same_mapping_if_no_overrides_specified(self):
        xml_mapping = {
            'any': {
                'tag1': 'xpath1'
            }
        }
        assert (
            get_xml_mapping_with_overrides(xml_mapping, None)
            == xml_mapping
        )

    def test_should_add_or_replace_property_in_multiple_top_level_keys(self):
        xml_mapping = {
            'top1': {
                'tag1': 'xpath1',
                'tag1.value': 'old'
            },
            'top2': {
                'tag1': 'xpath1',
                'tag1.value': 'old'
            }
        }
        xml_mapping_overrides = {
            'tag1.value': 'new',
            'tag1.extra': 'extra1'
        }
        expected_xml_mapping = {
            'top1': {
                'tag1': 'xpath1',
                'tag1.value': 'new',
                'tag1.extra': 'extra1'
            },
            'top2': {
                'tag1': 'xpath1',
                'tag1.value': 'new',
                'tag1.extra': 'extra1'
            }
        }
        assert (
            get_xml_mapping_with_overrides(xml_mapping, xml_mapping_overrides)
            == expected_xml_mapping
        )
