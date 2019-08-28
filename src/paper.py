from src.utility_functions import *
from copy import deepcopy
from nltk import word_tokenize, corpus, pos_tag
import re
remove_punct = re.compile("[^\w\s-]")
stop_words = corpus.stopwords.words("english")


class Paper:
    def __init__(self, pid, title, abstract, authors, unknown=None, affiliations=None, title_tokenized=None, title_pos=None):
        self.pid = pid
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.unknown = unknown if unknown else []
        self.affiliations = affiliations if affiliations else {}
        self.pid_sortable = convertPaperToSortable(pid)
        self.coauthors = {}
        if not title_tokenized or not title_pos:
            self.title_tokenized = []
            self.title_pos = []
        else:
            self.title_tokenized = title_tokenized
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
            "title_pos": self.title_pos
        }

    def copy(self, memodict={}):
        return Paper(**deepcopy(self.asDict()))

    def createPOS(self, remove_stops=True):
        title_tokenized = word_tokenize(remove_punct.sub(" ", cleanName(self.title,replace_punct=False)))
        title_pos = pos_tag(title_tokenized)
        for i, w in enumerate(title_tokenized):
            if w not in stop_words and remove_stops:
                self.title_tokenized.append(w)
            self.title_pos.append(title_pos[i][1])
