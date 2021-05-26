import logging
from itertools import groupby
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sciencebeam_alignment.levenshtein import get_levenshtein_ratio

from ..core.annotation.annotator import (
    AbstractAnnotator,
    Annotator
)

from ..core.structured_document import (
    split_tag_prefix,
    B_TAG_PREFIX
)

from ..structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    TeiText
)

from .target_annotation import TargetAnnotation


LOGGER = logging.getLogger(__name__)


def _iter_all_tokens(
        structured_document: GrobidTrainingTeiStructuredDocument) -> Iterable[TeiText]:
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )


def get_token_text(tokens: List[TeiText]) -> str:
    if not tokens:
        return ''
    return ''.join([
        text
        for token in tokens[:-1]
        for text in (token.text, token.whitespace or '')
    ] + [tokens[-1].text])


def get_target_annotation_name(target_annotation: TargetAnnotation) -> str:
    return target_annotation.name


def get_entities_name(entity_pair: Tuple[str, str]) -> str:
    return entity_pair[0]


def iter_structured_document_entities(
        structured_document: GrobidTrainingTeiStructuredDocument
        ) -> List[Tuple[str, str]]:
    pending_tokens = []
    pending_tag_value = None
    for token in _iter_all_tokens(structured_document):
        tag = structured_document.get_tag(token)
        tag_prefix, tag_value = split_tag_prefix(tag)
        if pending_tokens:
            if pending_tag_value != tag_value or tag_prefix == B_TAG_PREFIX:
                yield pending_tag_value, get_token_text(pending_tokens)
                pending_tokens = []
                pending_tag_value = None
        if not tag_value:
            continue
        pending_tag_value = tag_value
        pending_tokens.append(token)
    if pending_tokens:
        yield pending_tag_value, get_token_text(pending_tokens)


def get_required_target_annotation_by_name(
        target_annotations: List[TargetAnnotation],
        require_matching_fields: Set[str]
        ) -> Dict[str, List[TargetAnnotation]]:
    return {
        name: list(grouped_target_annotations)
        for name, grouped_target_annotations in groupby(
            sorted(target_annotations, key=get_target_annotation_name),
            key=get_target_annotation_name
        )
        if name in require_matching_fields
    }


def get_required_target_value_by_name(
        target_annotations: List[TargetAnnotation],
        require_matching_fields: Set[str]
        ) -> Dict[str, List[TargetAnnotation]]:
    result = {}
    required_target_annotation_by_name = get_required_target_annotation_by_name(
        target_annotations=target_annotations,
        require_matching_fields=require_matching_fields
    )
    for require_matching_field in require_matching_fields:
        required_target_annotations = required_target_annotation_by_name.get(
            require_matching_field
        )
        if not required_target_annotations:
            continue
        if len(required_target_annotations) != 1:
            raise RuntimeError(
                'only supporting single value fields, but found: %s' % required_target_annotations
            )
        required_value = required_target_annotations[0].value
        if not isinstance(required_value, str):
            raise RuntimeError(
                'only simple str required values supported, but found: %s'
                % required_target_annotations[0]
            )
        result[require_matching_field] = required_value
    return result


def get_structured_document_entities_by_name(
        structured_document: GrobidTrainingTeiStructuredDocument
        ) -> Dict[str, List[str]]:
    return {
        name: [pair[1] for pair in grouped_entities]
        for name, grouped_entities in groupby(
            sorted(
                iter_structured_document_entities(structured_document),
                key=get_entities_name
            ),
            key=get_entities_name
        )
    }


def is_structured_document_passing_checks(
        structured_document: GrobidTrainingTeiStructuredDocument,
        require_matching_fields: Optional[Set[str]],
        required_fields: Optional[Set[str]],
        target_annotations: List[TargetAnnotation],
        threshold: float = 0.8) -> bool:
    require_matching_fields = set(require_matching_fields or set()) | set(required_fields or set())
    if not require_matching_fields:
        return True
    if not target_annotations:
        raise RuntimeError('target_annotations required')
    required_value_by_name = get_required_target_value_by_name(
        target_annotations=target_annotations,
        require_matching_fields=require_matching_fields
    )
    LOGGER.debug('required_fields: %s', required_fields)
    if required_fields:
        missing_required_fields = set(required_fields) - set(required_value_by_name.keys())
        if missing_required_fields:
            LOGGER.warning('missing_required_fields: %s', missing_required_fields)
            return False
    if not required_value_by_name:
        return True
    entities_by_name = get_structured_document_entities_by_name(structured_document)
    LOGGER.info('entities_by_name: %s', entities_by_name)
    for require_matching_field, required_value in required_value_by_name.items():
        actual_entity_values = entities_by_name.get(require_matching_field, [])
        if not actual_entity_values:
            LOGGER.warning('required field not in tagged entities: %s', require_matching_field)
            return False
        actual_entity_joined_values = ' '.join(actual_entity_values)
        match_ratio = get_levenshtein_ratio(required_value, actual_entity_joined_values)
        if match_ratio < threshold:
            LOGGER.warning(
                'required field found, but not matching (%s): %r !~ %r',
                require_matching_field, required_value, actual_entity_joined_values
            )
            return False
    return True


def get_target_annotations_from_annotator(
        annotator: AbstractAnnotator) -> Optional[List[TargetAnnotation]]:
    # this is slightly hacky, we just want to get hold of the target annotations
    # which are hidden in one of the annotator
    if isinstance(annotator, Annotator):
        annotators = annotator.annotators
    else:
        annotators = [annotator]
    for _annotator in annotators:
        try:
            return _annotator.target_annotations  # type: ignore
        except AttributeError:
            pass
    return None
