import numpy as np
import shutil
from tqdm import tqdm
from scipy.special import comb
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from src.utility_functions import cleanName, convertPaperToSortable
from collections import Counter
from py_stringmatching.similarity_measure import soft_tfidf
from textdistance import JaroWinkler
from copy import deepcopy
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def getAlgo(algorithm="jaro", measure="similarity"):
    if algorithm == "jaro":
        algo = JaroWinkler()
    else:
        raise ValueError("Recieved invalid argument for algorithm")

    if measure == "similarity":
        return algo.similarity
    elif measure == "distance":
        return algo.distance
    else:
        raise ValueError("Recieved invalid argument for algorithm")


class CompareAuthors:
    """
    If you would like to implement your own author comparison, overwrite this class. Things you need for it to work:
        -- The class must be called CompareAuthors
        -- __init__ must have **kwargs or arguments with defaults
            -- These arguments passed will be initialized by preprocess_data.py and the configs/options you choose
        -- The only function called in CreateTrainingData is self.__call__()
            -- __call__ must have arguments key,tag,a,b
            -- __call__ must return key, tag, and an np array of the results
        -- It should (isn't required but recommended for testing) have a predefined compare_terms which maps to the
        returned np.array returned from __call__
        -- Any data you want to be passed in a,b is created in CreateTrainData.getAuthorInfo()
    """
    compare_terms = [
        "first_name_score",
        "initials_score",
        "org_name_score",
        "org_type_score",
        "email_domain_score",
        "co_auth_score",
        "co_auth_name1",
        "co_auth_email_avg",
        "co_auth_aff_avg",
        "co_auth_aff_type_score",
        "shared_aff_score",
        "shared_aff_type_score",
        "shared_aff_email",
        "department_score",
        "year_dif",
        "same_title_words",
        "venue",
        "num_citations_diff",
        "citation_auth_score",
        "citation_titles_score",
        "section_titles_scores",
        "post_code",
        "settlement",
        "country"
    ]

    def __init__(self, **kwargs):
        try:
            company_corpus = kwargs["company_corpus"]
        except KeyError as e:
            raise KeyError("company_corpus was not passed")
        try:
            department_corpus = kwargs["department_corpus"]
        except KeyError as e:
            raise KeyError("department_corpus was not passed")
        try:
            str_algorithm = kwargs["str_algorithm"]
        except KeyError as e:
            raise KeyError("str_algorithm was not passed")
        try:
            threshold = kwargs["threshold"]
        except KeyError as e:
            threshold = .5

        self.algorithm = getAlgo(*str_algorithm)
        self.org_name_algo = soft_tfidf.SoftTfIdf(corpus_list=company_corpus, threshold=threshold).get_raw_score
        self.dep_name_algo = soft_tfidf.SoftTfIdf(corpus_list=department_corpus, threshold=threshold).get_raw_score
        self.value_on_fail = 5

    def __call__(self, args):
        """
        Compare two authors, args MUST conatin:
        :param key: The key of the pair
        :type key: str
        :param tag: The tag associated with this pair (same or different
        :type tag: int
        :param a: author info for author a
        :type a: dict
        :param b: author info for author b
        :type b: dict
        :return: key, tag, np.array of result vector
        """
        key, tag, a, b = args
        name_a = a["name"].split(" ")
        name_b = b["name"].split(" ")
        try:
            initials_a = [x[0] for x in name_a]
        except:
            initials_a = []
        try:
            initials_b = [x[0] for x in name_b]
        except:
            initials_b = []
        len_name_a = len(name_a)
        len_name_b = len(name_b)
        full_name_count = 2
        if len_name_a >= full_name_count and len_name_b >= full_name_count:
            first_name_score = self.algorithm(name_a[0], name_b[0])
            last_name_score = self.algorithm(name_a[-1], name_b[-1])
            if len_name_a > full_name_count and len_name_b > full_name_count:
                middle_name_score = self.algorithm(" ".join(name_a[1:-1]), " ".join(name_b[1:-1]))
            elif len_name_a == full_name_count and len_name_b == full_name_count:
                middle_name_score = 1
            else:
                middle_name_score = 0
        else:
            # They both only have 1 name
            if len_name_a == len_name_b:
                first_name_score = 1
                middle_name_score = 1
                last_name_score = self.algorithm(name_a[0], name_b[0])
            else:
                first_name_score = 0
                middle_name_score = 0
                last_name_score = self.algorithm(name_a[-1], name_b[-1])
        shared_initials = 0
        len_ia = len(initials_a)
        len_ib = len(initials_b)
        for i in range(min(len_ia, len_ib)):
            if initials_a[i] == initials_b[i]:
                shared_initials += 1
        initials_score = shared_initials * min(len_ia, len_ib) / float(max(len_ia, len_ib))
        address_a = a["address"]
        address_b = b["address"]
        address_keys = ["postCode", "settlement", "country"]
        address_scores = []
        for k in address_keys:
            if k in address_a and k in address_b:
                if not address_a[k] and not address_b[k]:
                    address_scores.append(1)
                elif not address_b[k] or not address_a[k]:
                    address_scores.append(0)
                else:
                    address_scores.append(self.algorithm(address_a[k], address_b[k]))
            else:
                address_scores.append(0)
        if not a["aff_name"] or not b["aff_name"]:
            org_name_score = 0
        else:
            aff_a, aff_b = a["aff_name"].split(), b["aff_name"].split()
            try:
                org_name_score = self.org_name_algo(aff_a, aff_b)
            except:
                org_name_score = 0
        org_type_score = 1 if a["aff_type"] == b["aff_type"] else 0

        email_domain_score = 0
        if a["email_domain"] and b["email_domain"]:
            email_domain_score = self.algorithm(a["email_domain"], b["email_domain"])
        #
        # email_user_score = 0
        # if a["email_user"] and b["email_user"]:
        #     email_user_score = self.algorithm(a["email_user"], b["email_user"])

        # same co auth score

        a_co_auth_count = len(a["co_authors_name"])
        b_co_auth_count = len(b["co_authors_name"])
        if a_co_auth_count == 0:
            a_co_auth_count = 1
        if b_co_auth_count == 0:
            b_co_auth_count = 1

        co_auth_score = self._sharedInLists(a["co_authors_name"], b["co_authors_name"])
        a_co_auth_domains = [x[1] for x in a["co_authors_email"]]
        b_co_auth_domains = [x[1] for x in b["co_authors_email"]]
        a_co_auth_aff_split = [[stemmer.stem(w) for w in x.split()] if x else [] for x in a["co_authors_aff"]]
        b_co_auth_aff_split = [[stemmer.stem(w) for w in x.split()] if x else [] for x in b["co_authors_aff"]]
        co_auth_name_scores, co_auth_email_scores, co_auth_aff_scores = self._getCoAuthScores(
            [a["co_authors_name"], b["co_authors_name"]],
            [a_co_auth_domains, b_co_auth_domains],
            [a_co_auth_aff_split, b_co_auth_aff_split]
        )

        if not co_auth_email_scores:
            co_auth_email_scores = [0]
        co_auth_email_avg = np.mean(co_auth_email_scores)
        co_auth_email_median = np.median(co_auth_email_scores)
        if not a["co_authors_name"] and not b["co_authors_name"]:
            co_auth_aff_type_score = 0.0
            share_aff_type_score = 0
        elif not a["co_authors_name"] or not b["co_authors_name"]:
            co_auth_aff_type_score = self.value_on_fail
            share_aff_type_score = self.value_on_fail
        else:
            a_aff_type_counts = Counter(a["co_authors_aff_type"])
            b_aff_type_counts = Counter(b["co_authors_aff_type"])
            co_auth_aff_type_score = sum(
                [abs(a_aff_type_counts[x] / a_co_auth_count - b_aff_type_counts[x] / b_co_auth_count) for x in
                 a_aff_type_counts.keys() if x])

            # The value on fail *2 is so that if both fail, it does not give a 0 back as that could cause false positives
            a_same_aff_type = a_aff_type_counts[a["aff_type"]] if a["aff_type"] else self.value_on_fail*2
            b_same_aff_type = b_aff_type_counts[b["aff_type"]] if b["aff_type"] else self.value_on_fail
            share_aff_type_score = abs(a_same_aff_type / a_co_auth_count - b_same_aff_type / b_co_auth_count)

        if not co_auth_aff_scores:
            co_auth_aff_scores = [0]
        co_auth_aff_avg = np.mean(co_auth_aff_scores)

        if not co_auth_name_scores:
            top_five_names_scores = [0 for x in range(5)]
        else:
            top_five_names_scores = list(reversed(sorted(co_auth_name_scores)))
            if len(top_five_names_scores) < 5:
                top_five_names_scores.extend(
                    [max(top_five_names_scores) for x in range(5 - len(top_five_names_scores))])
        co_auth_name1, co_auth_name2, co_auth_name3, co_auth_name4, co_auth_name5 = top_five_names_scores[:5]

        shared_aff_score = self._getSharedScore(a["aff_name"], b["aff_name"],
                                                a["co_authors_aff"], b["co_authors_aff"],
                                                len(a["co_authors_name"]), len(b["co_authors_name"]))

        shared_aff_email = self._getSharedScore(a["email_domain"], b["email_domain"],
                                                a_co_auth_domains, b_co_auth_domains,
                                                len(a["co_authors_name"]), len(b["co_authors_name"]))

        # department score
        department_score = self._getDepartmentScore(a["department"], b["department"])

        # title score
        same_title_words = self._sharedInLists(a["title_tokenized"], b["title_tokenized"])

        # Year score
        year_dif = abs(convertPaperToSortable(a["pid"], True) - convertPaperToSortable(b["pid"], True))

        same_venue = 1 if a["pid"][0] == b["pid"][0] else 0

        a_citation_authors = []
        b_citation_authors = []
        for c in a["citations"]:
            a_citation_authors.extend(c["authors"])
        for c in b["citations"]:
            b_citation_authors.extend(c["authors"])
        num_citations_diff = abs(len(a_citation_authors)-len(b_citation_authors))
        citation_auth_score = self._sharedInLists(a_citation_authors,b_citation_authors)
        citation_titles_score = self._sharedInLists(a["citations_tokenized"],b["citations_tokenized"],size_mult=False)
        section_titles_scores = self._sharedInLists(a["sections_tokenized"],b["sections_tokenized"],size_mult=False)
        out = [
            first_name_score,
            initials_score,
            org_name_score,
            org_type_score,
            email_domain_score,
            co_auth_score,
            co_auth_name1,
            co_auth_email_avg,
            co_auth_aff_avg,
            co_auth_aff_type_score,
            shared_aff_score,
            share_aff_type_score,
            shared_aff_email,
            department_score,
            year_dif,
            same_title_words,
            same_venue,
            num_citations_diff,
            citation_auth_score,
            citation_titles_score,
            section_titles_scores
        ]
        out.extend(address_scores)
        return key, tag, np.asarray(out)

    @staticmethod
    def compareCoAuthValues(a_value, b_value, compare_algorithm):
        if not a_value or not b_value:
            return 0
        return compare_algorithm(a_value, b_value)

    def _getCoAuthScores(self, names, emails, affiliations):
        a_names, b_names = names
        a_emails, b_emails = emails
        a_orgs, b_orgs = affiliations

        name_scores = []
        email_scores = []
        org_scores = []
        for i in range(len(a_names)):
            for j in range(len(b_names)):
                name_scores.append(self.compareCoAuthValues(a_names[i].lower(), b_names[j].lower(), self.algorithm))
                email_scores.append(self.compareCoAuthValues(a_emails[i], b_emails[j], self.algorithm))
                org_scores.append(self.compareCoAuthValues(a_orgs[i], b_orgs[j], self.org_name_algo))

        if not email_scores:
            email_scores = [0]
        if not org_scores:
            org_scores = [0]
        return name_scores, email_scores, org_scores

    @staticmethod
    def _getSharedScore(a_value, b_value, a_co_auth, b_co_auth, a_len, b_len):
        if b_len == 0 and a_len == 0:
            return 0
        elif a_len == 0 or b_len == 0:
            return 10
        a_shared_aff = sum([1 for x in a_co_auth if x == a_value])
        b_shared_aff = sum([1 for x in b_co_auth if x == b_value])
        return abs(a_shared_aff / a_len - b_shared_aff / b_len)

    def _getDepartmentScore(self, a, b):
        scores = []
        if not a or not b:
            return 0
        for i in range(len(a)):
            for j in range(len(b)):
                dep_a = [stemmer.stem(w) for w in a[i].split()]
                dep_b = [stemmer.stem(w) for w in b[j].split()]
                try:
                    scores.append(self.dep_name_algo(dep_a, dep_b))
                except ZeroDivisionError as e:
                    scores.append(self.algorithm(a[i], b[j]))
        return max(scores)

    @staticmethod
    def _sharedInLists(a, b,size_mult=True):
        if not a or not b:
            return 0
        b_remaining = deepcopy(b)
        score = 0
        for i in a:
            if i in b_remaining:
                score += 1
                b_remaining.remove(i)
        if size_mult:
            return score * (min(len(a), len(b)) / float(max(len(a), len(b))))
        else:
            return score
    def processBatch(self, batch):
        out = []
        for i in batch:
            out.append(self(i))
        return out
