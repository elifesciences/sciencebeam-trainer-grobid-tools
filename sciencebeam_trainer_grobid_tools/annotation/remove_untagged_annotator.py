from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
)
from sciencebeam_gym.preprocess.annotation.annotator import (
    AbstractAnnotator
)


class RemoveUntaggedPostProcessingAnnotator(AbstractAnnotator):
    def annotate(self, structured_document: AbstractStructuredDocument):
        structured_document.remove_all_untagged()
        return structured_document
