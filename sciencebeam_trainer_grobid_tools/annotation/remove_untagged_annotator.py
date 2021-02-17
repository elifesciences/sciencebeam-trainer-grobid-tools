from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument,
)
from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)


class RemoveUntaggedPostProcessingAnnotator(AbstractAnnotator):
    def annotate(self, structured_document: AbstractStructuredDocument):
        structured_document.remove_all_untagged()
        return structured_document
