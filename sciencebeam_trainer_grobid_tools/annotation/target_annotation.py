import logging
import re

from lxml import etree

from sciencebeam_utils.utils.collection import (
    filter_truthy,
    strip_all
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    XmlMappingSuffix,
    TargetAnnotation,
    parse_xpaths,
    parse_json_with_default,
    re_compile_or_none,
    get_sub_mapping,
    extract_sub_annotations,
    match_xpaths,
    extract_children,
    get_stripped_text_content,
    apply_pattern,
    extract_using_regex,
    flatten_if_nested
)


LOGGER = logging.getLogger(__name__)


def is_blank(text: str) -> bool:
    return not text or not text.strip()


def contains_raw_text(element: etree.Element) -> bool:
    if not is_blank(element.text):
        return True
    for child in element:
        if not is_blank(child.tail):
            return True
    return False


def is_ends_with_word(text: str) -> bool:
    return re.match(r'.*\w$', text)


def is_starts_with_word(text: str) -> bool:
    return re.match(r'^\w.*', text)


def get_raw_text_content(element: etree.Element) -> str:
    text_list = []
    for text in element.itertext():
        if text_list and is_ends_with_word(text_list[-1]) and is_starts_with_word(text):
            text_list.append(' ')
        text_list.append(text)
    return ''.join(text_list)


def xml_root_to_target_annotations(xml_root, xml_mapping):
    if xml_root.tag not in xml_mapping:
        raise Exception("unrecognised tag: {} (available: {})".format(
            xml_root.tag, xml_mapping.sections())
        )

    mapping = xml_mapping[xml_root.tag]

    field_names = [k for k in mapping.keys() if '.' not in k]

    def get_mapping_flag(k, suffix):
        return mapping.get(k + suffix) == 'true'

    def get_match_multiple(k):
        return get_mapping_flag(k, XmlMappingSuffix.MATCH_MULTIPLE)

    def get_bonding_flag(k):
        return get_mapping_flag(k, XmlMappingSuffix.BONDING)

    def get_require_next_flag(k):
        return get_mapping_flag(k, XmlMappingSuffix.REQUIRE_NEXT)

    get_unmatched_parent_text_flag = (
        lambda k: get_mapping_flag(k, XmlMappingSuffix.UNMATCHED_PARENT_TEXT)
    )

    LOGGER.debug('fields: %s', field_names)

    target_annotations_with_pos = []
    xml_pos_by_node = {node: i for i, node in enumerate(xml_root.iter())}
    for k in field_names:
        match_multiple = get_match_multiple(k)
        bonding = get_bonding_flag(k)
        require_next = get_require_next_flag(k)
        unmatched_parent_text = get_unmatched_parent_text_flag(k)
        children_xpaths = parse_xpaths(mapping.get(k + XmlMappingSuffix.CHILDREN))
        children_concat = parse_json_with_default(
            mapping.get(k + XmlMappingSuffix.CHILDREN_CONCAT), []
        )
        children_range = parse_json_with_default(
            mapping.get(k + XmlMappingSuffix.CHILDREN_RANGE), []
        )
        re_compiled_pattern = re_compile_or_none(
            mapping.get(k + XmlMappingSuffix.REGEX)
        )
        extract_re_compiled_pattern = re_compile_or_none(
            mapping.get(k + XmlMappingSuffix.EXTRACT_REGEX)
        )
        LOGGER.debug('extract_re_compiled_pattern (%s): %s', k, extract_re_compiled_pattern)

        priority = int(mapping.get(k + XmlMappingSuffix.PRIORITY, '0'))
        sub_xpaths = get_sub_mapping(mapping, k)
        LOGGER.debug('sub_xpaths (%s): %s', k, sub_xpaths)

        xpaths = parse_xpaths(mapping[k])
        LOGGER.debug('xpaths(%s): %s', k, xpaths)
        for e in match_xpaths(xml_root, xpaths):
            e_pos = xml_pos_by_node.get(e)

            sub_annotations = extract_sub_annotations(e, sub_xpaths, mapping, k)
            LOGGER.debug('sub_annotations (%s): %s', k, sub_annotations)

            if children_xpaths and not contains_raw_text(e):
                text_content_list, standalone_values = extract_children(
                    e, children_xpaths, children_concat, children_range, unmatched_parent_text
                )
            else:
                text_content_list = filter_truthy(strip_all([get_raw_text_content(e)]))
                standalone_values = []
            if re_compiled_pattern:
                text_content_list = filter_truthy([
                    apply_pattern(s, re_compiled_pattern) for s in text_content_list
                ])
            if extract_re_compiled_pattern:
                text_content_list = filter_truthy([
                    extract_using_regex(s, extract_re_compiled_pattern) for s in text_content_list
                ])
            text_content_list = flatten_if_nested(text_content_list)
            if text_content_list:
                value = (
                    text_content_list[0]
                    if len(text_content_list) == 1
                    else sorted(text_content_list, key=lambda s: -len(s))
                )
                target_annotations_with_pos.append((
                    (-priority, e_pos),
                    TargetAnnotation(
                        value,
                        k,
                        match_multiple=match_multiple,
                        bonding=bonding,
                        require_next=require_next,
                        sub_annotations=sub_annotations
                    )
                ))
            if standalone_values:
                for i, standalone_value in enumerate(standalone_values):
                    target_annotations_with_pos.append((
                        (-priority, e_pos, i),
                        TargetAnnotation(
                            standalone_value,
                            k,
                            match_multiple=match_multiple,
                            bonding=bonding,
                            sub_annotations=sub_annotations
                        )
                    ))
    target_annotations_with_pos = sorted(
        target_annotations_with_pos,
        key=lambda x: x[0]
    )
    LOGGER.debug('target_annotations_with_pos:\n%s', target_annotations_with_pos)
    target_annotations = [
        x[1] for x in target_annotations_with_pos
    ]
    LOGGER.debug('target_annotations:\n%s', '\n'.join([
        ' ' + str(a) for a in target_annotations
    ]))
    return target_annotations
