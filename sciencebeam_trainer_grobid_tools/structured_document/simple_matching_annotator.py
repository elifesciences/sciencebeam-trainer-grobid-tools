import logging
import re
from distutils.util import strtobool
from itertools import groupby
from typing import Dict, List, Tuple

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)

from sciencebeam_gym.structured_document import (
    strip_tag_prefix,
    split_tag_prefix,
    add_tag_prefix,
    B_TAG_PREFIX,
    I_TAG_PREFIX
)

from sciencebeam_trainer_grobid_tools.utils.misc import get_safe

from sciencebeam_trainer_grobid_tools.utils.fuzzy import (
    fuzzy_search_index_range,
    iter_fuzzy_search_all_index_ranges
)

from sciencebeam_trainer_grobid_tools.structured_document.matching_utils import (
    SequenceWrapper,
    PendingSequences,
    SequencesText,
    join_tokens_text,
    normalise_and_remove_junk_str,
    normalise_str_or_list,
    normalise_and_remove_junk_str_or_list
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


DEFAULT_MERGE_ENABLED = True
DEFAULT_EXTEND_TO_LINE_ENABLED = True


class SimpleTagConfig:
    def __init__(
            self,
            match_prefix_regex: str = None,
            alternative_spellings: Dict[str, List[str]] = None,
            merge_enabled: bool = DEFAULT_MERGE_ENABLED,
            extend_to_line_enabled: bool = DEFAULT_EXTEND_TO_LINE_ENABLED,
            block_name: str = None):
        self.match_prefix_regex = match_prefix_regex
        self.alternative_spellings = alternative_spellings
        self.merge_enabled = merge_enabled
        self.extend_to_line_enabled = extend_to_line_enabled
        self.block_name = block_name

    def __repr__(self):
        return (
            '%s(match_prefix_regex=%s, alternative_spellings=%s,'
            + ' merge_enabled%s, extend_to_line_enabled=%s,'
            + ' block_name=%s)'
        ) % (
            type(self).__name__, self.match_prefix_regex, self.alternative_spellings,
            self.merge_enabled, self.extend_to_line_enabled,
            self.block_name
        )


DEFAULT_SIMPLE_TAG_CONFIG = SimpleTagConfig()


class SimpleSimpleMatchingConfig:
    def __init__(
            self,
            threshold: float = 0.8,
            lookahead_sequence_count: int = 200,
            min_token_length: int = 2,
            exact_word_match_threshold: int = 5,
            use_begin_prefix: bool = True,
            extend_to_line_enabled: bool = True,
            use_sub_annotations: bool = False,
            tag_config_map: Dict[str, SimpleTagConfig] = None):
        self.threshold = threshold
        self.lookahead_sequence_count = lookahead_sequence_count
        self.min_token_length = min_token_length
        self.exact_word_match_threshold = exact_word_match_threshold
        self.use_begin_prefix = use_begin_prefix
        self.extend_to_line_enabled = extend_to_line_enabled
        self.use_sub_annotations = use_sub_annotations
        self.tag_config_map = tag_config_map or {}

    def __repr__(self):
        return ''.join([
            '%s(threshold=%s,',
            ' lookahead_sequence_count=%s,',
            ' exact_word_match_threshold=%s,',
            ' use_begin_prefix=%s,',
            ' extend_to_line_enabled=%s,',
            ' use_sub_annotations=%s',
            ' tag_config_map=%s)'
         ]) % (
            type(self).__name__,
            self.threshold,
            self.lookahead_sequence_count,
            self.exact_word_match_threshold,
            self.use_begin_prefix,
            self.extend_to_line_enabled,
            self.use_sub_annotations,
            self.tag_config_map
        )

    def get_tag_config(self, tag_name: str) -> SimpleTagConfig:
        return self.tag_config_map.get(tag_name, DEFAULT_SIMPLE_TAG_CONFIG)


def merge_index_ranges(index_ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
    return (
        min(start for start, _ in index_ranges),
        max(end for _, end in index_ranges)
    )


def sorted_index_ranges(index_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(index_ranges)


def index_range_len(index_range: Tuple[int, int]) -> int:
    return index_range[1] - index_range[0]


class IndexRangeCluster:
    def __init__(self, index_ranges: List[Tuple[int, int]]):
        self.index_ranges = sorted_index_ranges(index_ranges)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.index_ranges)

    @property
    def start(self):
        return self.index_ranges[0][0]

    @property
    def end(self):
        return self.index_ranges[-1][1]

    @property
    def length(self):
        return self.end - self.start

    def gap_to(self, cluster: 'IndexRangeCluster') -> int:
        if cluster.start >= self.end:
            return cluster.start - self.end
        return self.start - cluster.end

    def should_merge(self, cluster: 'IndexRangeCluster') -> bool:
        gap = self.gap_to(cluster)
        max_length = max(self.length, cluster.length)
        if gap <= max_length + 10:
            return True
        return False

    def merge_with(self, cluster: 'IndexRangeCluster') -> 'IndexRangeCluster':
        return IndexRangeCluster(self.index_ranges + cluster.index_ranges)


def merge_related_clusters(clusters: List[IndexRangeCluster]) -> List[IndexRangeCluster]:
    while True:
        merged_clusters = [clusters[0]]
        unmerged_clusters = clusters[1:]
        has_merged = False
        for unmerged_cluster in unmerged_clusters:
            if merged_clusters[-1].should_merge(unmerged_cluster):
                merged_clusters[-1] = merged_clusters[-1].merge_with(unmerged_cluster)
                has_merged = True
            else:
                merged_clusters.append(unmerged_cluster)
        if not has_merged:
            return merged_clusters
        clusters = merged_clusters


def select_index_ranges(index_ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
    if len(index_ranges) <= 1:
        return index_ranges, []
    index_ranges = sorted_index_ranges(index_ranges)
    merged_clusters = merge_related_clusters([
        IndexRangeCluster([index_range])
        for index_range in index_ranges
    ])
    sorted_by_length_merged_clusters = sorted(
        merged_clusters, key=lambda cluster: cluster.length, reverse=True
    )
    selected = sorted_by_length_merged_clusters[0].index_ranges
    unselected = sorted([
        index_range
        for cluster in sorted_by_length_merged_clusters[1:]
        for index_range in cluster.index_ranges
    ])
    return selected, unselected


def _iter_all_lines(structured_document: AbstractStructuredDocument):
    return (
        line
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
    )


def to_begin_tag(tag: str) -> str:
    prefix, tag_value = split_tag_prefix(tag)
    return (
        add_tag_prefix(tag_value, prefix=B_TAG_PREFIX)
        if prefix == I_TAG_PREFIX
        else tag
    )


def to_inside_tag(tag: str) -> str:
    prefix, tag_value = split_tag_prefix(tag)
    return (
        add_tag_prefix(tag_value, prefix=I_TAG_PREFIX)
        if prefix == B_TAG_PREFIX
        else tag
    )


def to_begin_inside_tags(tag: str, length: int) -> List[str]:
    if not length:
        return []
    prefix, tag_value = split_tag_prefix(tag)
    if not prefix:
        return [tag] * length
    return (
        [add_tag_prefix(tag_value, prefix=B_TAG_PREFIX)] +
        [add_tag_prefix(tag_value, prefix=I_TAG_PREFIX)] * (length - 1)
    )


def get_merged_begin_inside_tags_of_same_tag_value(tags: List[str]) -> List[str]:
    if not tags:
        return []
    prefix, tag_value = split_tag_prefix(tags[0])
    if not prefix:
        return tags
    return (
        tags[:1] +
        [add_tag_prefix(tag_value, prefix=I_TAG_PREFIX)] * (len(tags) - 1)
    )


def get_extended_line_token_tags(
        line_token_tags: List[str],
        extend_to_line_enabled_map: Dict[str, bool] = None,
        merge_enabled_map: Dict[str, bool] = None,
        default_extend_to_line_enabled: bool = DEFAULT_EXTEND_TO_LINE_ENABLED,
        default_merge_enabled: bool = DEFAULT_MERGE_ENABLED) -> List[str]:
    if extend_to_line_enabled_map is None:
        extend_to_line_enabled_map = {}
    if merge_enabled_map is None:
        merge_enabled_map = {}
    LOGGER.debug(
        'line_token_tags: %s (extend_to_line_enabled_map: %s, merge_enabled_map: %s)',
        line_token_tags, extend_to_line_enabled_map, merge_enabled_map
    )
    grouped_token_tags = [
        list(group)
        for _, group in groupby(line_token_tags, key=strip_tag_prefix)
    ]
    grouped_token_tags = [
        (
            get_merged_begin_inside_tags_of_same_tag_value(group)
            if merge_enabled_map.get(strip_tag_prefix(group[0]), default_merge_enabled)
            else group
        )
        for group in grouped_token_tags
    ]
    LOGGER.debug('grouped_token_tags: %s', grouped_token_tags)
    result = []
    for index, group in enumerate(grouped_token_tags):
        prev_group = grouped_token_tags[index - 1] if index > 0 else None
        next_group = grouped_token_tags[index + 1] if index + 1 < len(grouped_token_tags) else None
        _, last_prev_tag_value = split_tag_prefix(get_safe(prev_group, -1))
        first_next_prefix, first_next_tag_value = split_tag_prefix(get_safe(next_group, 0))
        LOGGER.debug('group: %s', group)
        if group[0]:
            result.extend(group)
        elif prev_group and next_group:
            if (
                    last_prev_tag_value == first_next_tag_value
                    and merge_enabled_map.get(last_prev_tag_value, default_merge_enabled)
            ):
                result.extend([to_inside_tag(prev_group[-1])] * len(group))
                if first_next_prefix == B_TAG_PREFIX:
                    next_group[0] = to_inside_tag(next_group[0])
            else:
                result.extend(group)
        elif (
                prev_group and not extend_to_line_enabled_map.get(
                    last_prev_tag_value, default_extend_to_line_enabled
                )
        ):
            result.extend(group)
        elif next_group and not extend_to_line_enabled_map.get(
                first_next_tag_value, default_extend_to_line_enabled):
            result.extend(group)
        elif prev_group and len(prev_group) > len(group):
            result.extend([to_inside_tag(prev_group[-1])] * len(group))
        elif next_group and len(next_group) > len(group):
            result.extend(to_begin_inside_tags(next_group[0], len(group)))
            if first_next_prefix == B_TAG_PREFIX:
                next_group[0] = to_inside_tag(next_group[0])
        else:
            result.extend(group)
    LOGGER.debug('result: %s', result)
    return result


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
        LOGGER.debug('config: %s', config)
        self.merge_enabled_map = {
            tag: tag_confg.merge_enabled
            for tag, tag_confg in self.config.tag_config_map.items()
        }
        self.extend_to_line_enabled_map = {
            tag: tag_confg.extend_to_line_enabled
            for tag, tag_confg in self.config.tag_config_map.items()
        }

    def get_fuzzy_matching_index_range(
            self, haystack: str, needle, **kwargs):
        if len(needle) < self.config.min_token_length:
            return None
        target_value = normalise_str_or_list(needle)
        LOGGER.debug('target_value: %s', target_value)
        if len(target_value) < self.config.exact_word_match_threshold:
            # line feeds are currently not default separators for WordSequenceMatcher
            haystack = haystack.replace('\n', ' ')
        index_range = fuzzy_search_index_range(
            haystack, target_value,
            threshold=self.config.threshold,
            exact_word_match_threshold=self.config.exact_word_match_threshold,
            **kwargs
        )
        if index_range:
            return index_range
        target_value_reduced = split_and_join_with_space(
            normalise_and_remove_junk_str_or_list(needle)
        )
        LOGGER.debug('target_value_reduced: %s', target_value_reduced)
        return fuzzy_search_index_range(
            haystack, target_value_reduced,
            threshold=self.config.threshold,
            exact_word_match_threshold=self.config.exact_word_match_threshold,
            **kwargs
        )

    def get_fuzzy_matching_index_range_with_alternative_spellings(
            self,
            haystack: str,
            needle,
            alternative_spellings: Dict[str, List[str]],
            **kwargs):
        index_range = self.get_fuzzy_matching_index_range(haystack, needle, **kwargs)
        if index_range or not alternative_spellings:
            return index_range
        LOGGER.debug('alternative_spellings: %s', alternative_spellings)
        LOGGER.debug('not matching needle: [%s]', needle)
        matching_alternative_spellings = alternative_spellings.get(needle, [])
        LOGGER.debug('matching_alternative_spellings: %s', matching_alternative_spellings)
        for alternative_needle in matching_alternative_spellings:
            index_range = self.get_fuzzy_matching_index_range(
                haystack, alternative_needle, **kwargs
            )
            if index_range:
                LOGGER.debug('found alternative_needle: %s', alternative_needle)
                return index_range
        return None

    def apply_match_prefix_regex_to_index_range(
            self,
            text: SequencesText,
            index_range: Tuple[int, int],
            tag_name: str):
        tag_config = self.config.tag_config_map.get(
            tag_name,
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
        return start_index, end_index

    def update_annotation_for_index_range(
            self,
            structured_document: AbstractStructuredDocument,
            text: SequencesText,
            index_range: Tuple[int, int],
            tag_name: str):
        matching_tokens = list(text.iter_tokens_between(index_range))
        LOGGER.debug('setting matching_tokens to "%s": [%s]', tag_name, matching_tokens)
        LOGGER.debug(
            'setting matching text to "%s": [%s]',
            tag_name, join_tokens_text(matching_tokens)
        )
        untagged_matching_tokens = [
            token
            for token in matching_tokens
            if not structured_document.get_tag(token)
        ]
        for index, token in enumerate(untagged_matching_tokens):
            prefix = None
            if self.config.use_begin_prefix:
                prefix = B_TAG_PREFIX if index == 0 else I_TAG_PREFIX
            full_tag = add_tag_prefix(tag_name, prefix=prefix)
            structured_document.set_tag(token, full_tag)

    def process_sub_annotations(
            self,
            structured_document: AbstractStructuredDocument,
            text: SequencesText,
            index_range: Tuple[int, int],
            sub_annotations: List[TargetAnnotation]):
        if not sub_annotations:
            return
        LOGGER.debug('processing sub annotations: %s', sub_annotations)
        tokens = list(text.iter_tokens_between(index_range))
        LOGGER.debug('sub_tokens: %s', tokens)
        sub_text = SequencesText([SequenceWrapper(structured_document, tokens)])
        sub_text_str = str(sub_text)
        LOGGER.debug('sub_text_str: %r', sub_text_str)
        for sub_annotation in sub_annotations:
            sub_tag_name = sub_annotation.name
            target_value = sub_annotation.value
            assert not isinstance(target_value, list), 'list sub annotation values not supported'
            sub_index_ranges_iterable = iter_fuzzy_search_all_index_ranges(
                sub_text_str, target_value,
                threshold=self.config.threshold,
                exact_word_match_threshold=self.config.exact_word_match_threshold
            )
            for sub_index_range in sub_index_ranges_iterable:
                LOGGER.debug(
                    'sub_annotation match: sub_tag=%r, value=%r sub_index_range=%s',
                    sub_tag_name, target_value, sub_index_range
                )
                matching_tokens = list(sub_text.iter_tokens_between(sub_index_range))
                LOGGER.debug(
                    'setting sub matching_tokens to "%s": %s',
                    sub_tag_name, matching_tokens
                )
                existing_matching_sub_tags = [
                    structured_document.get_sub_tag(token)
                    for token in matching_tokens
                ]
                if any(existing_matching_sub_tags):
                    LOGGER.debug('some tokens already have sub tags, skipping')
                    continue
                for index, token in enumerate(matching_tokens):
                    prefix = None
                    if self.config.use_begin_prefix:
                        prefix = B_TAG_PREFIX if index == 0 else I_TAG_PREFIX
                    full_tag = add_tag_prefix(sub_tag_name, prefix=prefix)
                    structured_document.set_sub_tag(token, full_tag)
                # accept the index range and move to next sub tag
                break
            LOGGER.debug(
                'sub_annotation match not found: sub_tag=%r, value=%r',
                sub_tag_name, target_value
            )

    def iter_matching_index_ranges(
            self,
            text: SequencesText,
            target_annotations: List[TargetAnnotation]):
        for target_annotation in target_annotations:
            LOGGER.debug('target_annotation: %s', target_annotation)
            LOGGER.debug('target_annotation.value: %s', target_annotation.value)
            tag_config = self.config.tag_config_map.get(target_annotation.name)
            alternative_spellings = tag_config and tag_config.alternative_spellings
            LOGGER.debug('alternative_spellings: %s', alternative_spellings)
            text_str = str(text)
            LOGGER.debug('text: %s', text)
            index_range = None
            if isinstance(target_annotation.value, list):
                index_ranges = [
                    self.get_fuzzy_matching_index_range_with_alternative_spellings(
                        text_str,
                        value,
                        alternative_spellings=alternative_spellings
                    )
                    for value in target_annotation.value
                ]
                matching_index_ranges = [
                    _index_range
                    for _index_range in index_ranges
                    if _index_range
                ]
                if matching_index_ranges:
                    selected_index_ranges, unselected_index_ranges = (
                        select_index_ranges(matching_index_ranges)
                    )
                    index_range = merge_index_ranges(selected_index_ranges)
                    LOGGER.debug(
                        'merged multi-value selected index ranges: %s -> %s',
                        text.get_index_ranges_with_text(selected_index_ranges), index_range
                    )
                    LOGGER.debug(
                        'merged multi-value unselected index ranges: %s',
                        text.get_index_ranges_with_text(unselected_index_ranges)
                    )
            else:
                index_range = self.get_fuzzy_matching_index_range_with_alternative_spellings(
                    text_str,
                    target_annotation.value,
                    alternative_spellings=alternative_spellings
                )
            LOGGER.debug('index_range: %s', index_range)
            if index_range:
                yield index_range

    def extend_annotations_to_whole_line(self, structured_document: AbstractStructuredDocument):
        for line in _iter_all_lines(structured_document):
            tokens = structured_document.get_tokens_of_line(line)
            line_token_tags = [structured_document.get_tag(token) for token in tokens]
            extended_line_token_tags = get_extended_line_token_tags(
                line_token_tags,
                extend_to_line_enabled_map=self.extend_to_line_enabled_map,
                merge_enabled_map=self.merge_enabled_map
            )
            LOGGER.debug(
                'line_token_tags, transformed: %s -> %s (tokens: %s)',
                line_token_tags, extended_line_token_tags, tokens
            )
            for token, token_tag in zip(tokens, extended_line_token_tags):
                if not token_tag:
                    continue
                structured_document.set_tag(token, token_tag)
        return structured_document

    def annotate(self, structured_document: AbstractStructuredDocument):
        pending_sequences = PendingSequences.from_structured_document(
            structured_document,
            normalize_fn=normalise_and_remove_junk_str
        )
        current_pending_sequences = pending_sequences
        target_annotations_grouped_by_tag = groupby(
            self.target_annotations,
            key=lambda target_annotation: target_annotation.name
        )
        current_block_name = None
        for tag_name, grouped_target_annotations in target_annotations_grouped_by_tag:
            tag_block_name = self.config.get_tag_config(tag_name).block_name or 'default'
            LOGGER.debug(
                'tag_block_name: %s (current_block_name: %s)',
                tag_block_name, current_block_name
            )
            grouped_target_annotations = list(grouped_target_annotations)
            LOGGER.debug('grouped_target_annotations: %s', grouped_target_annotations)
            for target_annotation in grouped_target_annotations:
                text = SequencesText(current_pending_sequences.get_pending_sequences(
                    limit=self.config.lookahead_sequence_count
                ))
                index_ranges = list(self.iter_matching_index_ranges(
                    text,
                    [target_annotation]
                ))
                if not index_ranges and current_block_name != tag_block_name:
                    LOGGER.debug(
                        'block name has changed, scanning whole document (%s)',
                        tag_block_name
                    )
                    text = SequencesText(pending_sequences.get_pending_sequences(
                        limit=None
                    ))
                    index_ranges = list(self.iter_matching_index_ranges(
                        text,
                        [target_annotation]
                    ))
                    if not index_ranges:
                        continue
                    index_range = merge_index_ranges(index_ranges)
                    block_index_range = (index_range[0], text.end_index)
                    current_pending_sequences = PendingSequences(
                        list(text.iter_sequences_between(block_index_range))
                    )
                    LOGGER.debug(
                        'set current_pending_sequences to %s (%s), text:\n%s',
                        block_index_range,
                        tag_block_name,
                        SequencesText(current_pending_sequences.get_pending_sequences())
                    )
                    current_block_name = tag_block_name
                if not index_ranges:
                    continue
                index_range = merge_index_ranges(index_ranges)
                LOGGER.debug('merged index ranges: %s -> %s', index_ranges, index_range)
                index_range = self.apply_match_prefix_regex_to_index_range(
                    text, index_range, tag_name
                )
                self.update_annotation_for_index_range(
                    structured_document,
                    text,
                    index_range,
                    tag_name
                )
                if self.config.use_sub_annotations:
                    self.process_sub_annotations(
                        structured_document,
                        text,
                        index_range,
                        sub_annotations=target_annotation.sub_annotations
                    )
        if self.config.extend_to_line_enabled:
            self.extend_annotations_to_whole_line(structured_document)
        return structured_document


class SimpleTagConfigProps:
    MATCH_PREFIX_REGEX = 'match-prefix-regex'
    ALTERNATIVE_SPELLINGS = 'alternative-spellings'
    MERGE = 'merge'
    EXTEND_TO_LINE = 'extend-to-line'
    BLOCK = 'block'


def parse_regex(regex_str: str) -> str:
    LOGGER.debug('regex_str: %s', regex_str)
    if not regex_str:
        return regex_str
    if len(regex_str) >= 2 and regex_str.startswith('"') and regex_str.endswith('"'):
        regex_str = regex_str[1:-1]
    # validate pattern
    re.compile(regex_str)
    LOGGER.debug('parsed regex_str: %s', regex_str)
    return regex_str


def parse_alternative_spellings(alternative_spellings_str: str) -> Dict[str, List[str]]:
    LOGGER.debug('alternative_spellings_str: %s', alternative_spellings_str)
    if not alternative_spellings_str:
        return {}
    result = {}
    for line in alternative_spellings_str.splitlines():
        line = line.strip()
        if not line:
            continue
        LOGGER.debug('line: %s', line)
        key, value_str = line.split('=', maxsplit=1)
        result[key.strip()] = value_str.strip().split(',')
    LOGGER.debug('alternative_spellings: %s', result)
    return result


def get_simple_tag_config(config_map: Dict[str, str], field: str) -> SimpleTagConfig:
    return SimpleTagConfig(
        match_prefix_regex=parse_regex(config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.MATCH_PREFIX_REGEX)
        )),
        alternative_spellings=parse_alternative_spellings(config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.ALTERNATIVE_SPELLINGS)
        )),
        merge_enabled=strtobool(config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.MERGE),
            str(DEFAULT_MERGE_ENABLED)
        )) == 1,
        extend_to_line_enabled=strtobool(config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.EXTEND_TO_LINE),
            str(DEFAULT_EXTEND_TO_LINE_ENABLED)
        )) == 1,
        block_name=config_map.get(
            '%s.%s' % (field, SimpleTagConfigProps.BLOCK)
        )
    )


def get_simple_tag_config_map(
        xml_mapping: Dict[str, Dict[str, str]]) -> Dict[str, SimpleTagConfig]:
    LOGGER.debug('xml_mapping: %s', xml_mapping)
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
