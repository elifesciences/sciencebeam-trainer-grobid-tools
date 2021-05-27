from typing import Any, Iterable

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument
)
from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)


def iter_all_tokens_excluding_space(
        structured_document: AbstractStructuredDocument) -> Iterable[Any]:
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )


def iter_all_tokens_including_space(
        structured_document: AbstractStructuredDocument) -> Iterable[Any]:
    assert isinstance(structured_document, GrobidTrainingTeiStructuredDocument)
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_all_tokens_of_line(line)
    )
