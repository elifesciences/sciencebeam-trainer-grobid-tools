import logging
import re
from typing import Dict, List, Tuple

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
    normalise_and_remove_junk_str,
    normalise_str_or_list,
    normalise_and_remove_junk_str_or_list
)

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    fuzzy_match
)

from sciencebeam_trainer_grobid_tools.structured_document.matching_utils import (
    PendingSequences,
    SequencesText
)


LOGGER = logging.getLogger(__name__)


def split_and_join_with_space(text: str) -> str:
    """
    Splits the given string and joins with space to reproduce the output of document tokens
    """
    return ' '.join([
        token
        for token in re.split(r'(\W)', text)
        if token.strip()
    ])


class SimpleTagConfig:
    def __init__(self, match_prefix_regex: str = None):
        self.match_prefix_regex = match_prefix_regex

    def __repr__(self):
        return '%s(match_prefix_regex=%s)' % (
            type(self).__name__, self.match_prefix_regex
        )


DEFAULT_SIMPLE_TAG_CONFIG = SimpleTagConfig()


class SimpleSimpleMatchingConfig:
    def __init__(
            self,
            threshold: float = 0.8,
            lookahead_sequence_count: int = 200,
            tag_config_map: Dict[str, SimpleTagConfig] = None):
        self.threshold = threshold
        self.lookahead_sequence_count = lookahead_sequence_count
        self.tag_config_map = tag_config_map or {}

    def __repr__(self):
        return '%s(threshold=%s, lookahead_sequence_count=%s, tag_config_map=%s)' % (
            type(self).__name__, self.threshold, self.lookahead_sequence_count, self.tag_config_map
        )


class SimpleMatchingAnnotator(AbstractAnnotator):
    """
    The SimpleMatchingAnnotator assumes that the lines are in the correct reading order.
    It doesn't implement all of the features provide by MatchingAnnotator.
    """
    def __init__(
            self,
            target_annotations: List[TargetAnnotation],
            config: SimpleSimpleMatchingConfig = None,
            **kwargs):
        self.target_annotations = target_annotations
        if config is None:
            config = SimpleSimpleMatchingConfig(**kwargs)
        elif kwargs:
            raise ValueError('either config or kwargs should be specified')
        self.config = config
        LOGGER.info('config: %s', config)

    def is_target_annotation_supported(self, target_annotation: TargetAnnotation) -> bool:
        if target_annotation.match_multiple:
            return False
        if target_annotation.bonding:
            return False
        if target_annotation.require_next:
            return False
        if target_annotation.sub_annotations:
            return False
        return True

    def get_fuzzy_matching_index_range(
            self, haystack: str, needle, **kwargs):
        target_value = split_and_join_with_space(
            normalise_str_or_list(needle)
        )
        LOGGER.debug('target_value: %s', target_value)
        fm = fuzzy_match(haystack, target_value, exact_word_match_threshold=5, **kwargs)
        LOGGER.debug('fm: %s', fm)
        if fm.b_gap_ratio() >= self.config.threshold:
            return fm.a_index_range()
        target_value_reduced = split_and_join_with_space(
            normalise_and_remove_junk_str_or_list(needle)
        )
        LOGGER.debug('target_value_reduced: %s', target_value_reduced)
        fm = fuzzy_match(haystack, target_value_reduced, exact_word_match_threshold=5, **kwargs)
        if fm.b_gap_ratio() >= self.config.threshold:
            return fm.a_index_range()
        return None

    def update_annotation_for_index_range(
            self,
            structured_document: AbstractStructuredDocument,
            text: SequencesText,
            index_range: Tuple[int, int],
            target_annotation: TargetAnnotation):
        tag_config = self.config.tag_config_map.get(
            target_annotation.name,
            DEFAULT_SIMPLE_TAG_CONFIG
        )
        start_index, end_index = index_range
        LOGGER.debug('index_range: %s', (start_index, end_index))
        LOGGER.debug('match_prefix_regex: [%s]', tag_config.match_prefix_regex)
        if start_index > 0 and tag_config.match_prefix_regex:
            prefix = str(text)[:start_index]
            LOGGER.debug('prefix: [%s]', prefix)
            m = re.search(tag_config.match_prefix_regex, prefix)
            if m:
                LOGGER.debug('match: [%s]', m.span())
                start_index = m.start()
        matching_tokens = list(text.iter_tokens_between((start_index, end_index)))
        LOGGER.debug('matching_tokens: %s', matching_tokens)
        for token in matching_tokens:
            if not structured_document.get_tag(token):
                structured_document.set_tag(token, target_annotation.name)

    def annotate(self, structured_document: AbstractStructuredDocument):
        pending_sequences = PendingSequences.from_structured_document(
            structured_document,
            normalize_fn=normalise_and_remove_junk_str
        )
        for target_annotation in self.target_annotations:
            LOGGER.debug('target_annotation: %s', target_annotation)
            # if not self.is_target_annotation_supported(target_annotation):
            #     raise NotImplementedError('unsupported target annotation: %s' % target_annotation)
            LOGGER.info('target_annotation.value: %s', target_annotation.value)
            text = SequencesText(pending_sequences.get_pending_sequences(
                limit=self.config.lookahead_sequence_count
            ))
            text_str = str(text)
            LOGGER.debug('text: %s', text)
            if isinstance(target_annotation.value, list):
                index_ranges = [
                    self.get_fuzzy_matching_index_range(text_str, value)
                    for value in target_annotation.value
                ]
                if all(index_ranges):
                    index_range = (
                        min(start for start, _ in index_ranges),
                        max(end for _, end in index_ranges)
                    )
            else:
                index_range = self.get_fuzzy_matching_index_range(text_str, target_annotation.value)
            LOGGER.debug('index_range: %s', index_range)
            if index_range:
                self.update_annotation_for_index_range(
                    structured_document,
                    text,
                    index_range,
                    target_annotation
                )
        return structured_document


class SimpleTagConfigProps:
    MATCH_PREFIX_REGEX = 'match-prefix-regex'


def get_simple_tag_config(config_map: Dict[str, str], field: str) -> SimpleTagConfig:
    return SimpleTagConfig(
        match_prefix_regex=config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.MATCH_PREFIX_REGEX)
        )
    )


def get_simple_tag_config_map(xml_mapping: Dict[str, Dict[str, str]]):
    LOGGER.info('xml_mapping: %s', xml_mapping)
    fields = {
        key
        for _, section_config_map in xml_mapping.items()
        for key in section_config_map.keys()
        if '.' not in key
    }
    flat_config_map = {
        key: value
        for _, section_config_map in xml_mapping.items()
        for key, value in section_config_map.items()
    }
    return {
        field: get_simple_tag_config(flat_config_map, field)
        for field in fields
    }
