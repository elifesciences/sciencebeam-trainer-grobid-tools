from sciencebeam_trainer_grobid_tools.core.structured_document import (
    AbstractStructuredDocument,
)
from sciencebeam_trainer_grobid_tools.core.annotation.annotator import (
    AbstractAnnotator
)
from sciencebeam_trainer_grobid_tools.structured_document.grobid_training_tei import (
    GrobidTrainingTeiStructuredDocument
)


class RemoveUntaggedPostProcessingAnnotator(AbstractAnnotator):
    def annotate(self, structured_document: AbstractStructuredDocument):
        assert isinstance(structured_document, GrobidTrainingTeiStructuredDocument)
        structured_document.remove_all_untagged()
        return structured_document
