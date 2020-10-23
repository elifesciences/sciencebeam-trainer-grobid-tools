import logging
from collections import Counter
from functools import partial
from typing import Callable, List, Set

from sciencebeam_gym.structured_document import (
    strip_tag_prefix
)

from ..structured_document.grobid_training_tei import (
    SUB_LEVEL,
    load_grobid_training_tei_structured_document,
    save_grobid_training_tei_structured_document,
    GrobidTrainingTeiStructuredDocument
)


LOGGER = logging.getLogger(__name__)


def _iter_all_tokens(structured_document):
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_all_tokens_of_line(line)
    )


def _map_token_tags(structured_document, tag_fn, **kwargs):
    for token in _iter_all_tokens(structured_document):
        tag = structured_document.get_tag_or_preserved_tag(token, **kwargs)
        updated_tag = tag_fn(tag, token)
        if updated_tag != tag:
            structured_document.set_tag_only(token, updated_tag, **kwargs)
            structured_document.clear_preserved_tag_only(token, **kwargs)


def _preserve_tag_fn(
        existing_tag: str,
        token,  # pylint: disable=unused-argument
        include_fields: List[str] = None,
        exclude_fields: List[str] = None):
    simple_existing_tag = strip_tag_prefix(existing_tag)
    if exclude_fields and simple_existing_tag in exclude_fields:
        return None
    if include_fields and simple_existing_tag not in include_fields:
        return None
    return existing_tag


def _no_preserve_tag_fn(*_):
    return None


def _get_used_sub_tag_counts(
        structured_document: GrobidTrainingTeiStructuredDocument) -> Counter:
    return Counter((
        structured_document.get_tag_or_preserved_tag_value(token, level=SUB_LEVEL)
        for token in _iter_all_tokens(structured_document)
    ))


def annotate_structured_document_inplace(
        structured_document: GrobidTrainingTeiStructuredDocument,
        annotator,
        preserve_tags: bool,
        fields: List[str],
        preserve_fields: List[str] = None,
        preserve_sub_tags: bool = False,
        no_preserve_sub_fields: Set[str] = None):
    if not fields:
        fields = set()
    if preserve_tags or preserve_fields:
        exclude_fields = set(fields) - set(preserve_fields or [])
        LOGGER.debug(
            'preserving tags, including %s, except for fields: %s',
            preserve_fields, exclude_fields
        )
        tag_fn = partial(
            _preserve_tag_fn,
            include_fields=preserve_fields,
            exclude_fields=exclude_fields
        )
    else:
        LOGGER.debug('not preserving tags')
        tag_fn = _no_preserve_tag_fn

    _map_token_tags(structured_document, tag_fn)

    if not preserve_sub_tags:
        LOGGER.debug('not preserving sub tags')
        _map_token_tags(structured_document, _no_preserve_tag_fn, level=SUB_LEVEL)
    elif no_preserve_sub_fields:
        LOGGER.debug(
            'preserving sub tags, except for fields: %s',
            no_preserve_sub_fields
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                'used sub tags counts: %s', _get_used_sub_tag_counts(structured_document)
            )
        _map_token_tags(
            structured_document,
            partial(
                _preserve_tag_fn,
                exclude_fields=no_preserve_sub_fields
            ),
            level=SUB_LEVEL
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                'used sub tags counts (after): %s', _get_used_sub_tag_counts(structured_document)
            )
    else:
        LOGGER.debug('preserving all sub tags')

    annotator.annotate(structured_document)


def _apply_preserved_fields(
        structured_document: GrobidTrainingTeiStructuredDocument,
        always_preserve_fields: Set[str]):
    num_tokens = 0
    all_preserved_tags = []
    for token in _iter_all_tokens(structured_document):
        full_preserved_tag = structured_document.get_tag_or_preserved_tag(token)
        all_preserved_tags.append(full_preserved_tag)
        preserved_tag = strip_tag_prefix(full_preserved_tag)
        if preserved_tag in always_preserve_fields:
            LOGGER.debug('apply preserved field: %s -> %s', token, full_preserved_tag)
            structured_document.set_tag(token, full_preserved_tag)
            num_tokens += 1
    LOGGER.debug(
        'applied preserved fields (%d tokens, all counts: %s): %s',
        num_tokens,
        Counter(all_preserved_tags),
        always_preserve_fields
    )


def annotate_structured_document(
        source_structured_document_path: str,
        target_structured_document_path: str,
        annotator,
        preserve_tags: bool,
        fields: List[str],
        always_preserve_fields: List[str] = None,
        preserve_sub_tags: bool = False,
        no_preserve_sub_fields: Set[str] = None,
        is_structured_document_passing_checks: Callable[
            [GrobidTrainingTeiStructuredDocument], bool
        ] = None,
        failed_target_structured_document_path: str = None,
        **kwargs):
    LOGGER.info('loading from: %s', source_structured_document_path)
    structured_document = load_grobid_training_tei_structured_document(
        source_structured_document_path,
        **kwargs
    )

    if always_preserve_fields:
        _apply_preserved_fields(structured_document, set(always_preserve_fields))

    annotate_structured_document_inplace(
        structured_document,
        annotator=annotator,
        preserve_tags=preserve_tags,
        preserve_fields=always_preserve_fields,
        preserve_sub_tags=preserve_sub_tags,
        no_preserve_sub_fields=no_preserve_sub_fields,
        fields=fields
    )

    if not is_structured_document_passing_checks(structured_document):
        if not failed_target_structured_document_path:
            LOGGER.warning(
                'document failed checks, skipping: %s', source_structured_document_path
            )
            return
        LOGGER.info('failed checks, saving to: %s', failed_target_structured_document_path)
        save_grobid_training_tei_structured_document(
            failed_target_structured_document_path,
            structured_document
        )
        return

    LOGGER.info('saving to: %s', target_structured_document_path)
    save_grobid_training_tei_structured_document(
        target_structured_document_path,
        structured_document
    )
