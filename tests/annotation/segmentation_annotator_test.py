import logging
from pathlib import Path
from typing import List, Tuple

from lxml.builder import E

from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument,
    ContainerNodePaths,
    add_tag_prefix,
    B_TAG_PREFIX,
    I_TAG_PREFIX
)

from sciencebeam_trainer_grobid_tools.annotation.segmentation_annotator import (
    parse_segmentation_config,
    is_valid_page_header_candidate,
    SegmentationConfig,
    SegmentationAnnotator,
    PageTagNames,
    FrontTagNames,
    BodyTagNames,
    BackTagNames,
    SegmentationTagNames
)


LOGGER = logging.getLogger(__name__)


SEGMENTATION_CONTAINER_NODE_PATH = ContainerNodePaths.SEGMENTATION_CONTAINER_NODE_PATH


DEFAULT_CONFIG = SegmentationConfig({
    SegmentationTagNames.FRONT: {FrontTagNames.TITLE, FrontTagNames.ABSTRACT},
    SegmentationTagNames.BODY: {BodyTagNames.SECTION_TITLE},
    SegmentationTagNames.REFERENCE: {BackTagNames.REFERENCE},
    SegmentationTagNames.ANNEX: {BackTagNames.APPENDIX}
})


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'
TOKEN_4 = 'token4'
TOKEN_5 = 'token5'
TOKEN_6 = 'token6'

PAGE_HEADER_TOKEN_1 = 'Page_Header_1'

LONG_PAGE_HEADER_TEXT_1 = 'This is a very long page header'

OTHER_TAG = 'other'


def _tei(items: list = None):
    return E.tei(E.text(
        *(items or [])
    ))


def _simple_document_with_tagged_token_lines(
        lines: List[List[Tuple[str, str]]]) -> GrobidTrainingTeiStructuredDocument:
    tei_items = []
    for line in lines:
        tei_items.append(' '.join(token for _, token in line))
        tei_items.append(E.lb())
    doc = GrobidTrainingTeiStructuredDocument(
        _tei(tei_items),
        container_node_path=SEGMENTATION_CONTAINER_NODE_PATH
    )
    doc_lines = [line for page in doc.get_pages() for line in doc.get_lines_of_page(page)]
    for line, doc_line in zip(lines, doc_lines):
        for (tag, token), doc_token in zip(line, doc.get_tokens_of_line(doc_line)):
            assert token == doc.get_text(doc_token)
            if tag:
                doc.set_tag(doc_token, tag)
    return doc


def _get_document_tagged_token_lines(
        doc: GrobidTrainingTeiStructuredDocument) -> List[List[Tuple[str, str]]]:
    document_tagged_token_lines = [
        [
            (doc.get_tag(token), doc.get_text(token))
            for token in doc.get_tokens_of_line(line)
        ]
        for page in doc.get_pages()
        for line in doc.get_lines_of_page(page)
    ]
    LOGGER.debug('document_tagged_token_lines: %s', document_tagged_token_lines)
    return document_tagged_token_lines


class TestParseSegmentationConfig:
    def test_should_parse_config(self, temp_dir: Path):
        config_path = temp_dir.joinpath('segmentation.conf')
        config_path.write_text('\n'.join([
            '[tags]',
            'front = title, abstract '
        ]))
        config = parse_segmentation_config(config_path)
        LOGGER.debug('config: %s', config)
        assert config.segmentation_mapping['front'] == {'title', 'abstract'}

    def test_should_parse_front_max_start_line_index(self, temp_dir: Path):
        config_path = temp_dir.joinpath('segmentation.conf')
        config_path.write_text('\n'.join([
            '[tags]',
            'front = title,abstract',
            '[config]',
            'front_max_start_line_index = 123 '
        ]))
        config = parse_segmentation_config(config_path)
        LOGGER.debug('config: %s', config)
        assert config.front_max_start_line_index == 123


class TestIsValidPageHeaderCandidate:
    def test_should_not_accept_all_digits(self):
        assert is_valid_page_header_candidate(
            '123',
            100
        ) is False

    def test_should_not_accept_all_digits_with_dot(self):
        assert is_valid_page_header_candidate(
            '123.45',
            100
        ) is False

    def test_should_not_accept_all_digits_with_space(self):
        assert is_valid_page_header_candidate(
            '123 45',
            100
        ) is False

    def test_should_not_accept_single_token_text(self):
        assert is_valid_page_header_candidate(
            'ThisIsALongSingleToken',
            100
        ) is False

    def test_should_accept_long_text(self):
        assert is_valid_page_header_candidate(
            LONG_PAGE_HEADER_TEXT_1,
            100
        ) is True

    def test_should_accept_long_text_starting_with_digit(self):
        assert is_valid_page_header_candidate(
            '123 ' + LONG_PAGE_HEADER_TEXT_1,
            100
        ) is True

    def test_should_not_accept_long_text_if_below_min_count(self):
        assert is_valid_page_header_candidate(
            '123 ' + LONG_PAGE_HEADER_TEXT_1,
            100,
            min_count=101
        ) is False


class TestSegmentationAnnotator:
    def test_should_not_fail_on_empty_document(self):
        structured_document = GrobidTrainingTeiStructuredDocument(
            _tei()
        )
        SegmentationAnnotator(DEFAULT_CONFIG).annotate(structured_document)

    def test_should_annotate_title_as_front(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1)
            ]
        ]

    def test_should_annotate_reference_as_reference(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(BackTagNames.REFERENCE, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.REFERENCE, TOKEN_1)
            ]
        ]

    def test_should_merge_separate_reference_if_enabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_1),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_2)
            ],
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_3),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_4)
            ]
        ])

        SegmentationAnnotator(
            DEFAULT_CONFIG._replace(no_merge_references=False)
        ).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.REFERENCE, TOKEN_1),
                (SegmentationTagNames.REFERENCE, TOKEN_2)
            ],
            [
                (SegmentationTagNames.REFERENCE, TOKEN_3),
                (SegmentationTagNames.REFERENCE, TOKEN_4)
            ]
        ]

    def test_should_keep_separate_reference_if_disabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_1),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_2)
            ],
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_3),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_4)
            ]
        ])

        SegmentationAnnotator(
            DEFAULT_CONFIG._replace(no_merge_references=True)
        ).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (add_tag_prefix(SegmentationTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_1),
                (add_tag_prefix(SegmentationTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_2)
            ],
            [
                (add_tag_prefix(SegmentationTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_3),
                (add_tag_prefix(SegmentationTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_4)
            ]
        ]

    def test_should_merge_and_fill_gap_between_reference_if_enabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_1),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_2)
            ],
            [
                (None, TOKEN_3),
                (None, TOKEN_4)
            ],
            [
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=B_TAG_PREFIX), TOKEN_5),
                (add_tag_prefix(BackTagNames.REFERENCE, prefix=I_TAG_PREFIX), TOKEN_6)
            ]
        ])

        SegmentationAnnotator(
            DEFAULT_CONFIG._replace(no_merge_references=False)
        ).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.REFERENCE, TOKEN_1),
                (SegmentationTagNames.REFERENCE, TOKEN_2)
            ],
            [
                (SegmentationTagNames.REFERENCE, TOKEN_3),
                (SegmentationTagNames.REFERENCE, TOKEN_4)
            ],
            [
                (SegmentationTagNames.REFERENCE, TOKEN_5),
                (SegmentationTagNames.REFERENCE, TOKEN_6)
            ]
        ]

    def test_should_merge_and_fill_gap_between_back_tags_if_enabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (add_tag_prefix(BackTagNames.APPENDIX, prefix=B_TAG_PREFIX), TOKEN_1),
                (add_tag_prefix(BackTagNames.APPENDIX, prefix=I_TAG_PREFIX), TOKEN_2)
            ],
            [
                (None, TOKEN_3),
                (None, TOKEN_4)
            ],
            [
                (add_tag_prefix(BackTagNames.APPENDIX, prefix=B_TAG_PREFIX), TOKEN_5),
                (add_tag_prefix(BackTagNames.APPENDIX, prefix=I_TAG_PREFIX), TOKEN_6)
            ]
        ])

        SegmentationAnnotator(
            DEFAULT_CONFIG._replace(no_merge_references=False)
        ).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.ANNEX, TOKEN_1),
                (SegmentationTagNames.ANNEX, TOKEN_2)
            ],
            [
                (SegmentationTagNames.ANNEX, TOKEN_3),
                (SegmentationTagNames.ANNEX, TOKEN_4)
            ],
            [
                (SegmentationTagNames.ANNEX, TOKEN_5),
                (SegmentationTagNames.ANNEX, TOKEN_6)
            ]
        ]

    def test_should_annotate_other_tags_as_body(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.BODY, TOKEN_1)
            ]
        ]

    def test_should_annotate_no_tag_as_body_if_preserve_is_disabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=False).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.BODY, TOKEN_1)
            ]
        ]

    def test_should_annotate_not_no_tag_as_body_if_preserve_is_enabled(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (None, TOKEN_1)
            ]
        ]

    def test_should_annotate_not_fail_on_empty_line(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [],
            [(None, TOKEN_1)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [],
            [(None, TOKEN_1)]
        ]

    def test_should_annotate_title_line_as_front(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (FrontTagNames.TITLE, TOKEN_1),
                (FrontTagNames.TITLE, TOKEN_2),
                (FrontTagNames.TITLE, TOKEN_3)
            ]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1),
                (SegmentationTagNames.FRONT, TOKEN_2),
                (SegmentationTagNames.FRONT, TOKEN_3)
            ]
        ]

    def test_should_annotate_line_with_using_common_tag(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [
                (FrontTagNames.TITLE, TOKEN_1),
                (FrontTagNames.TITLE, TOKEN_2),
                (OTHER_TAG, TOKEN_3)
            ]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [
                (SegmentationTagNames.FRONT, TOKEN_1),
                (SegmentationTagNames.FRONT, TOKEN_2),
                (SegmentationTagNames.FRONT, TOKEN_3)
            ]
        ]

    def test_should_annotate_untagged_lines_between_first_and_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(None, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_annotate_untagged_lines_before_first_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_not_annotate_untagged_lines_after_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(None, TOKEN_3)],
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(None, TOKEN_3)]
        ]

    def test_should_not_annotate_untagged_page_no_lines_between_first_and_last_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(PageTagNames.PAGE, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.PAGE, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_clear_minority_among_untagged_tag(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1), (None, TOKEN_2), (OTHER_TAG, TOKEN_3)]
        ])

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(None, TOKEN_1), (None, TOKEN_2), (None, TOKEN_3)]
        ]

    def test_should_ignore_front_if_start_line_index_beyond_threshold(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, TOKEN_1)],
            [(None, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        config = SegmentationConfig(
            DEFAULT_CONFIG.segmentation_mapping,
            front_max_start_line_index=1
        )
        SegmentationAnnotator(config, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(None, TOKEN_1)],
            [(None, TOKEN_2)],
            [(None, TOKEN_3)]
        ]

    def test_should_keep_front_if_line_index_of_front_started_before_threshold(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(FrontTagNames.TITLE, TOKEN_3)]
        ])

        config = SegmentationConfig(
            DEFAULT_CONFIG.segmentation_mapping,
            front_max_start_line_index=1
        )
        SegmentationAnnotator(config, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.FRONT, TOKEN_3)]
        ]

    def test_should_annotate_page_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(None, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(FrontTagNames.ABSTRACT, TOKEN_2)],
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.HEADNOTE, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.HEADNOTE, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(SegmentationTagNames.FRONT, TOKEN_2)]
        ]

    def test_should_annotate_assume_front_or_body_after_page_header(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(None, TOKEN_1)],
            [(FrontTagNames.TITLE, TOKEN_2)],
            [(None, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(None, TOKEN_3)],
            [(BodyTagNames.SECTION_TITLE, TOKEN_4)],
        ])

        SegmentationAnnotator(DEFAULT_CONFIG).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.HEADNOTE, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.FRONT, TOKEN_2)],
            [(SegmentationTagNames.HEADNOTE, t) for t in LONG_PAGE_HEADER_TEXT_1.split(' ')],
            [(SegmentationTagNames.BODY, TOKEN_3)],
            [(SegmentationTagNames.BODY, TOKEN_4)]
        ]

    def test_should_not_annotate_preserved_page_numbers_as_headnote(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, '1')],
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(None, '1')],
            [(BodyTagNames.SECTION_TITLE, TOKEN_2)],
        ])
        all_tokens = list(doc.iter_all_tokens())
        doc._set_preserved_tag(all_tokens[0], PageTagNames.PAGE)  # pylint: disable=protected-access
        doc._set_preserved_tag(all_tokens[2], PageTagNames.PAGE)  # pylint: disable=protected-access

        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.PAGE, '1')],
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.PAGE, '1')],
            [(SegmentationTagNames.BODY, TOKEN_2)]
        ]

    def test_should_find_missing_page_numbers_annotations(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, '1')],
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(PageTagNames.PAGE, '2')],
            [(BodyTagNames.SECTION_TITLE, TOKEN_2)],
            [(PageTagNames.PAGE, '3')]
        ])
        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.PAGE, '1')],
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.PAGE, '2')],
            [(SegmentationTagNames.BODY, TOKEN_2)],
            [(SegmentationTagNames.PAGE, '3')]
        ]

    def test_should_not_annotate_out_of_order_page_number(self):
        doc = _simple_document_with_tagged_token_lines(lines=[
            [(None, '2')],
            [(FrontTagNames.TITLE, TOKEN_1)],
            [(PageTagNames.PAGE, '2')],
            [(BodyTagNames.SECTION_TITLE, TOKEN_2)],
            [(PageTagNames.PAGE, '3')]
        ])
        SegmentationAnnotator(DEFAULT_CONFIG, preserve_tags=True).annotate(doc)
        assert _get_document_tagged_token_lines(doc) == [
            [(SegmentationTagNames.FRONT, '2')],
            [(SegmentationTagNames.FRONT, TOKEN_1)],
            [(SegmentationTagNames.PAGE, '2')],
            [(SegmentationTagNames.BODY, TOKEN_2)],
            [(SegmentationTagNames.PAGE, '3')]
        ]
