import re
from typing import List

from sciencebeam_trainer_grobid_tools.core.structured_document import (
    SimpleStructuredDocument,
    SimplePage,
    SimpleLine,
    SimpleToken,
    B_TAG_PREFIX,
    I_TAG_PREFIX,
    add_tag_prefix
)


def get_token_texts_for_text(text: str) -> List[str]:
    return [s for s in re.split(r'(\W)', text) if s.strip()]


def get_entity_tokens(tag: str, value: str) -> List[SimpleToken]:
    return [
        SimpleToken(token_text, tag=add_tag_prefix(
            tag,
            prefix=B_TAG_PREFIX if index == 0 else I_TAG_PREFIX
        ))
        for index, token_text in enumerate(get_token_texts_for_text(value))
    ]


class SimpleDocumentBuilder:
    def __init__(self):
        self.doc = SimpleStructuredDocument(lines=[])

    @property
    def current_page(self) -> SimplePage:
        return self.doc._pages[-1]  # pylint: disable=protected-access

    def get_or_create_current_line(self) -> SimpleLine:
        lines = self.current_page.lines
        if not lines:
            lines.append(SimpleLine([]))
        return lines[-1]

    def write_tokens(self, tokens: List[SimpleToken]):
        line = self.get_or_create_current_line()
        line.tokens.extend(tokens)
        return self

    def write_entity(self, tag: str, value: str):
        self.write_tokens(get_entity_tokens(tag, value))
        return self
