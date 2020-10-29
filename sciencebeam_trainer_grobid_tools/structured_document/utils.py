from typing import Any, Iterable

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument
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
    return (
        token
        for page in structured_document.get_pages()
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_all_tokens_of_line(line)
    )
