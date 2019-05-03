from sciencebeam_trainer_grobid_tools.utils.regex import (
    regex_change_name
)


class TestRegexChangeName(object):
    def test_should_change_matching_name(self):
        assert regex_change_name(
            'file1-suffix.tei.xml',
            r'/(.*)-suffix.*/\1.xml.gz/'
        ) == 'file1.xml.gz'
