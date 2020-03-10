from typing import Dict

import pytest
from lxml.builder import E

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    parse_xml_mapping
)

from sciencebeam_trainer_grobid_tools.annotation.target_annotation import (
    xml_root_to_target_annotations
)

from sciencebeam_trainer_grobid_tools.auto_annotate_utils import (
    DEFAULT_ANNOT_CONFIG_FILENAME
)

from tests.auto_annotate_test_utils import (
    get_target_xml_node
)


@pytest.fixture(name='xml_mapping')
def _xml_mapping() -> Dict[str, Dict[str, str]]:
    return parse_xml_mapping('config/%s' % DEFAULT_ANNOT_CONFIG_FILENAME)


class TestAnnotXmlFrontConf:
    def test_should_extract_author_names(self, xml_mapping: Dict[str, Dict[str, str]]):
        xml_root = get_target_xml_node(
            author_nodes=[
                E.contrib(E.name(
                    E.surname('Maison'),
                    E('given-names', 'Mary')
                )),
                E.contrib(E.name(
                    E.surname('Smith'),
                    E('given-names', 'John')
                ))
            ]
        )
        target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
        assert [
            (target_annotation.name, target_annotation.value)
            for target_annotation in target_annotations
        ] == [
            ('author', ['Maison', 'Mary']),
            ('author', ['Smith', 'John'])
        ]

    def test_should_extract_author_aff_within_author_aff(
            self, xml_mapping: Dict[str, Dict[str, str]]):
        xml_root = get_target_xml_node(
            author_nodes=[
                E.contrib(
                    E.name(
                        E.surname('Smith'),
                        E('given-names', 'John')
                    ),
                    E.aff(
                        E.institution('University of Science'),
                        E.country('Smithonia')
                    )
                )
            ]
        )
        target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
        assert [
            (target_annotation.name, target_annotation.value)
            for target_annotation in target_annotations
        ] == [
            ('author', ['Smith', 'John']),
            ('author_aff', ['University of Science', 'Smithonia'])
        ]

    def test_should_extract_raw_author_aff_if_containing_raw_text(
            self, xml_mapping: Dict[str, Dict[str, str]]):
        xml_root = get_target_xml_node(
            author_nodes=[
                E.contrib(
                    E.name(
                        E.surname('Smith'),
                        E('given-names', 'John')
                    ),
                    E.aff(
                        E.label('1'),
                        E.institution('University of Science'),
                        ', Smithonia'
                    )
                )
            ]
        )
        target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
        assert [
            (target_annotation.name, target_annotation.value)
            for target_annotation in target_annotations
        ] == [
            ('author', ['Smith', 'John']),
            ('author_aff', '1 University of Science, Smithonia')
        ]

    def test_should_extract_author_aff_address_line(
            self, xml_mapping: Dict[str, Dict[str, str]]):
        xml_root = get_target_xml_node(
            author_nodes=[
                E.contrib(
                    E.name(
                        E.surname('Smith'),
                        E('given-names', 'John')
                    ),
                    E.aff(
                        E.institution('University of Science'),
                        E.country('Smithonia'),
                        E('addr-line', E('named-content', 'East St'))
                    )
                )
            ]
        )
        target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
        assert [
            (target_annotation.name, target_annotation.value)
            for target_annotation in target_annotations
        ] == [
            ('author', ['Smith', 'John']),
            ('author_aff', ['University of Science', 'Smithonia', 'East St'])
        ]

    def test_should_extract_author_aff_email(
            self, xml_mapping: Dict[str, Dict[str, str]]):
        xml_root = get_target_xml_node(
            author_nodes=[
                E.contrib(
                    E.name(
                        E.surname('Smith'),
                        E('given-names', 'John')
                    ),
                    E.aff(
                        E.institution('University of Science'),
                        E.email('john@smiths.test')
                    )
                )
            ]
        )
        target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
        assert [
            (target_annotation.name, target_annotation.value)
            for target_annotation in target_annotations
        ] == [
            ('author', ['Smith', 'John']),
            ('author_aff', ['University of Science', 'john@smiths.test']),
            ('email', 'john@smiths.test')
        ]
