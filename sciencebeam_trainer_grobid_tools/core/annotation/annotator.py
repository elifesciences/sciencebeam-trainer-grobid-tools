from abc import ABC, abstractmethod
from typing import List


class AbstractAnnotator(ABC):
    @abstractmethod
    def annotate(self, structured_document):
        pass


DEFAULT_ANNOTATORS: List[AbstractAnnotator] = []


class Annotator:
    def __init__(self, annotators: List[AbstractAnnotator] = None):
        if annotators is None:
            annotators = DEFAULT_ANNOTATORS
        self.annotators = annotators

    def annotate(self, structured_document):
        for annotator in self.annotators:
            structured_document = annotator.annotate(structured_document)
        return structured_document
