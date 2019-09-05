import logging
from typing import List

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    TargetAnnotation
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
    normalise_and_remove_junk_str,
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


class SimpleSimpleMatchingConfig:
    def __init__(self, threshold: float = 0.8, lookahead_sequence_count: int = 200):
        self.threshold = threshold
        self.lookahead_sequence_count = lookahead_sequence_count


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

    def annotate(self, structured_document: AbstractStructuredDocument):
        pending_sequences = PendingSequences.from_structured_document(
            structured_document,
            normalize_fn=normalise_and_remove_junk_str
        )
        for target_annotation in self.target_annotations:
            LOGGER.debug('target_annotation: %s', target_annotation)
            if not self.is_target_annotation_supported(target_annotation):
                raise NotImplementedError('unsupported target annotation: %s' % target_annotation)
            target_value = normalise_and_remove_junk_str_or_list(target_annotation.value)
            LOGGER.debug('target_value: %s', target_value)
            # pending sequences provides a view of the not yet untagged tokens
            # this is what we will try to align the target value to
            text = SequencesText(pending_sequences.get_pending_sequences(
                limit=self.config.lookahead_sequence_count
            ))
            LOGGER.debug('text: %s', text)
            fm = fuzzy_match(str(text), target_value, exact_word_match_threshold=5)
            LOGGER.debug('fm: %s', fm)
            if fm.b_gap_ratio() >= self.config.threshold:
                index_range = fm.a_index_range()
                LOGGER.debug('index_range: %s', index_range)
                matching_tokens = list(text.iter_tokens_between(index_range))
                LOGGER.debug('matching_tokens: %s', matching_tokens)
                for token in matching_tokens:
                    if not structured_document.get_tag(token):
                        structured_document.set_tag(token, target_annotation.name)
        return structured_document
