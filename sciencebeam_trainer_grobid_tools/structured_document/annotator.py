import logging
from functools import partial


from .grobid_training_tei import (
    load_grobid_training_tei_structured_document,
    save_grobid_training_tei_structured_document,
    DEFAULT_CONTAINER_NODE_PATH
)


def get_logger():
    return logging.getLogger(__name__)


def _iter_all_tokens(structured_document):
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_all_tokens_of_line(line)
    )


def _map_token_tags(structured_document, tag_fn):
    for token in _iter_all_tokens(structured_document):
        tag = structured_document.get_tag_or_preserved_tag(token)
        updated_tag = tag_fn(tag, token)
        if updated_tag != tag:
            structured_document.set_tag(token, updated_tag)


def _preserve_tag_fn(existing_tag, token, fields, structured_document):
    if existing_tag not in fields and structured_document.get_text(token).strip():
        return existing_tag
    return None


def _no_preserve_tag_fn(*_):
    return None


def annotate_structured_document_inplace(
        structured_document,
        annotator,
        preserve_tags,
        fields):
    if not fields:
        fields = set()
    if preserve_tags:
        get_logger().debug('preserving tags, except for fields: %s', fields)
        tag_fn = partial(
            _preserve_tag_fn,
            fields=fields,
            structured_document=structured_document
        )
    else:
        get_logger().debug('not preserving tags')
        tag_fn = _no_preserve_tag_fn

    _map_token_tags(structured_document, tag_fn)

    annotator.annotate(structured_document)


def annotate_structured_document(
        source_structured_document_path: str,
        target_structured_document_path: str,
        annotator,
        preserve_tags: bool,
        fields,
        container_node_path: str = DEFAULT_CONTAINER_NODE_PATH):
    get_logger().info('loading from: %s', source_structured_document_path)
    structured_document = load_grobid_training_tei_structured_document(
        source_structured_document_path,
        container_node_path=container_node_path
    )

    annotate_structured_document_inplace(
        structured_document,
        annotator=annotator,
        preserve_tags=preserve_tags,
        fields=fields
    )

    get_logger().info('saving to: %s', target_structured_document_path)
    save_grobid_training_tei_structured_document(
        target_structured_document_path,
        structured_document
    )
