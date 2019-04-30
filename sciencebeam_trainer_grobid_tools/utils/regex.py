import logging
import re
from collections import namedtuple


LOGGER = logging.getLogger(__name__)


SubstitutionPattern = namedtuple('SubstitutionPattern', [
    'match_pattern',
    'replace_pattern',
    'suffix',
    'options'
])


def _parse_subsitution_pattern(regex_pattern):
    LOGGER.debug('regex_pattern: %s', regex_pattern)
    delimiter = regex_pattern[0]
    fragments = regex_pattern.split(delimiter)
    LOGGER.debug('fragments: %s', fragments)
    suffix, match_pattern, replace_pattern, options = fragments
    return SubstitutionPattern(
        suffix=suffix,
        match_pattern=match_pattern,
        replace_pattern=replace_pattern,
        options=options
    )


def regex_change_name(name, regex_pattern):
    substitution_pattern = _parse_subsitution_pattern(regex_pattern)
    return re.sub(
        substitution_pattern.match_pattern,
        substitution_pattern.replace_pattern,
        name
    )
