from abc import ABC, abstractmethod


class AbstractAnnotator(ABC):
    @abstractmethod
    def annotate(self, structured_document):
        pass


DEFAULT_ANNOTATORS = []


class Annotator:
    def __init__(self, annotators=None):
        if annotators is None:
            annotators = DEFAULT_ANNOTATORS
        self.annotators = annotators

    def annotate(self, structured_document):
        for annotator in self.annotators:
            structured_document = annotator.annotate(structured_document)
        return structured_document
