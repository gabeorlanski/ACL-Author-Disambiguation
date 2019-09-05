from src.utility_functions import *
from copy import deepcopy
from nltk import word_tokenize, corpus, pos_tag
import re
from src.utility_functions import removeDupes

remove_punct = re.compile("[^\w\s-]")
stop_words = corpus.stopwords.words("english")


class Paper:
    def __init__(self, pid, title, abstract, authors, unknown=None, affiliations=None, title_tokenized=None,
                 sections=None, sections_tokenized=None, citations=None, citations_tokenized=None, title_pos=None):

        self.pid = pid
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.unknown = unknown if unknown else []
        self.affiliations = affiliations if affiliations else {}
        self.pid_sortable = convertPaperToSortable(pid)
        self.coauthors = {}
        if not title_tokenized:
            self.title_tokenized = []
        else:
            self.title_tokenized = title_tokenized
        if not citations:
            self.citations = []
            self.citations_tokenized = []
        else:
            self.citations = citations
            self.citations_tokenized = citations_tokenized

        if sections is None:
            self.sections = {}
            self.sections_tokenized = []
        else:
            self.sections = sections
            self.sections_tokenized = sections_tokenized if sections_tokenized else []
        self.title_pos = title_pos

    def addAffiliations(self, affiliations):
        if affiliations:
            raise ValueError("affiliations is not empty")
        self.affiliations = deepcopy(affiliations)

    def __hash__(self):
        return hash(self.pid)

    def __eq__(self, other):
        if self.pid != other.pid:
            return False
        if self.title != other.title:
            return False
        return True

    def __ne__(self, other):
        return not self == other


    def isAuthor(self, author) -> bool:
        return author in self.authors

    def asDict(self) -> dict:
        return {
            "pid": self.pid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "unknown": self.unknown,
            "affiliations": self.affiliations,
            "title_tokenized": self.title_tokenized,
            "citations": self.citations,
            "citations_tokenized": self.citations_tokenized,
            "sections": self.sections,
            "sections_tokenized": self.sections_tokenized
        }

    def copy(self):
        return Paper(**deepcopy(self.asDict()))

    def loadTokenized(self, title, citations, sections):
        self.title_tokenized = title
        self.citations_tokenized = citations
        self.sections_tokenized = sections

    def tokenize(self, remove_stops=True):
        title_tokenized = word_tokenize(remove_punct.sub(" ", cleanName(self.title, replace_punct=False)))
        title_out = []
        for i, w in enumerate(title_tokenized):
            if (w not in stop_words and remove_stops) or not remove_stops:
                title_out.append(w)
        citation_titles = [word_tokenize(x["title"]) for x in self.citations if x["title"]]
        citations_out = []
        for c in citation_titles:
            for i, w in enumerate(c):
                if (w not in stop_words and remove_stops) or not remove_stops:
                    citations_out.append(w)
        section_words = []
        for k, info in self.sections.items():
            for n, sent in info.items():
                section_words.extend(word_tokenize(sent.lower()))
        section_words = list(set(section_words))
        section_out = [w for w in section_words if (w not in stop_words and remove_stops) or not remove_stops]

        return removeDupes(title_tokenized), removeDupes(citations_out), removeDupes(section_out)
