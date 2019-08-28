from unittest import TestCase
import json
from src.compare_authors import CompareAuthors
import warnings
from py_stringmatching.similarity_measure import soft_tfidf
from textdistance import JaroWinkler
from copy import deepcopy


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestCompareAuthors(TestCase):
    @ignore_warnings
    def setUp(self) -> None:
        self.test_auth_info = json.load(open("/home/gabe/Desktop/research-main/tests/createPairTests/test_papers.json"))
        self.org_corpus= [x.strip() for x in open("/home/gabe/Desktop/research-main/data/txt/org_corpus.txt").readlines()]
        self.department_corpus = [x.strip() for x in open(
            "/home/gabe/Desktop/research-main/data/txt/department_corpus.txt").readlines()]
        self.org_algo = soft_tfidf.SoftTfIdf(self.org_corpus,threshold=.4).get_raw_score
        self.dep_algo = soft_tfidf.SoftTfIdf(self.department_corpus,threshold=.4).get_raw_score

    @staticmethod
    def compareResultsToDict(r):
        out = {}
        for i, v in enumerate(CompareAuthors.compare_terms):
            try:
                out[v] = r[i]
            except IndexError as e:
                raise Exception("Lengths are mismatched: {} vs {}".format(len(r), len(CompareAuthors.compare_terms)))
        return out

    def checkCompareResults(self,actual,expected):
        missing = []
        wrong_values = []
        remaining_keys = list(actual.keys())
        for k in expected.keys():
            try:
                if expected[k] != actual[k]:
                    wrong_values.append([k,actual[k],expected[k]])
                remaining_keys.remove(k)
            except KeyError as e:
                missing.append(k)
        self.assertEqual(missing,[])
        self.assertEqual(remaining_keys,[])
        self.assertEqual(wrong_values,[])

    def test_call(self):
        algorithm = JaroWinkler().similarity
        a = self.test_auth_info["P19-1642 iacer-calixto"]
        a["department"] = ["Department of Biomedical Informatics and Medical Education"]
        b = self.test_auth_info["C16-1050 elaheh-shafieibavani"]
        b["department"] = ["Center for Information"
                           "Department of Computing",
                           "Institute of Biomedical Engineering"]
        tmp_dep_scores = []
        for i in a["department"]:
            for j in b["department"]:
                tmp_dep_scores.append(self.dep_algo(i.split(),j.split()))
        dep_score = max(tmp_dep_scores)
        aff_iacer = "ILLC The University of Amsterdam".split()
        aff_elaheh = "University of New South Wales".split()
        test_1 = {
            "first_name_score": algorithm("Iacer", "Elaheh"),
            # "middle_name_score": 1.0,
            # "last_name_score": algorithm("Calixto", "ShafieiBavani"),
            "initials_score": 0.0,
            "org_name_score": self.org_algo(aff_iacer, aff_elaheh),
            "org_type_score": 1.0,
            "email_domain_score": algorithm("uva.nl", "cse.unsw.edu.au"),
            # "email_user_score": algorithm("iacer.calixto", "elahehs"),
            "co_auth_score": 0.0,
            "co_auth_name1": 0.5162210338680927,
            # "co_auth_name2": 0.449554367201426,
            # "co_auth_name3": 0.44949494949494956,
            # "co_auth_name4":  0.42424242424242414,
            # "co_auth_name5": 0.39646464646464646,
            "co_auth_email_avg": 0.15185185185185185,
            # "co_auth_email_median": 0.0,
            "co_auth_aff_avg": 0,
            # "co_auth_aff_median": 0,
            "co_auth_aff_type_score": 0.0,
            "shared_aff_score": 0.0,
            "shared_aff_type_score": 0.0,
            "shared_aff_email": abs(.5 - 2 / 3),
            "department_score": dep_score,
            "same_title_words": 0.0,
            "venue": 0,
            "year_dif": 3.0,
            # "post_code": 0.0,
            # "settlement": 0.0,
            # "country": 0.0,

        }
        test_2 = {
            "first_name_score": algorithm("Iacer", "Elaheh"),
            # "middle_name_score": 1.0,
            # "last_name_score": algorithm("Calixto", "ShafieiBavani"),
            "initials_score": 0.0,
            "org_name_score": self.org_algo(aff_iacer, aff_elaheh),
            "org_type_score": 1.0,
            "email_domain_score": algorithm("uva.nl", "cse.unsw.edu.au"),
            # "email_user_score": algorithm("iacer.calixto", "elahehs"),
            "co_auth_score": 0.0,
            "co_auth_name1": 0.0,
            # "co_auth_name2": 0.0,
            # "co_auth_name3": 0.0,
            # "co_auth_name4": 0.0,
            # "co_auth_name5": 0.0,
            "co_auth_email_avg": 0.0,
            # "co_auth_email_median": 0.0,
            "co_auth_aff_avg": 0,
            # "co_auth_aff_median": 0,
            "co_auth_aff_type_score": 5.0,
            "shared_aff_score": 10.0,
            "shared_aff_type_score": 5.0,
            "shared_aff_email": 10.0,
            "department_score": dep_score,
            "same_title_words": 0.0,
            "venue": 0,
            "year_dif": 3.0,
            # "post_code": 0.0,
            # "settlement": 0.0,
            # "country": 0.0,

        }

        c = deepcopy(b)
        c["co_authors_name"] =[]
        arg_t1= ["", 0, a, b]
        arg_t2=  ["", 0, a, c]
        comparator = CompareAuthors(company_corpus=self.org_corpus,department_corpus=self.department_corpus,
                                  str_algorithm=["jaro", "similarity"])

        k, t, res = comparator(arg_t1)
        self.assertEqual(k, arg_t1[0])
        self.assertEqual(t, arg_t1[1])
        self.checkCompareResults(self.compareResultsToDict(res),test_1)

        k, t, res = comparator(arg_t2)
        self.assertEqual(k, arg_t2[0])
        self.assertEqual(t, arg_t2[1])
        self.checkCompareResults(self.compareResultsToDict(res), test_2)

        # print("Average runtime to compare authors {:.3E}s".format(np.mean(run_times)))

