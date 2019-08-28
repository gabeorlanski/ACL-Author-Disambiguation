from unittest import TestCase
from src.compare_authors import CompareAuthors
from src.utility_functions import *
from src.create_training_data import getAuthorInfo
from src.author_disambiguation import AuthorDisambiguation
import logging
import os
import warnings
import json
from src.paper import Paper
from nltk.stem import PorterStemmer
from copy import deepcopy
import random

stemmer = PorterStemmer()
os.chdir("..")


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestAuthorDisambiguation(TestCase):

    @ignore_warnings
    def setUp(self) -> None:
        test_papers = json.load(open(os.getcwd() + "/tests/authorDisambiguationTests/test_papers.json"))
        self.test_papers = {}
        for k, p in test_papers.items():
            self.test_papers[k] = Paper(**p)
        self.config = json.load(open(os.getcwd() + "/config.json"))
        data_path = os.getcwd() + "/data"
        # papers_dict = json.load(open(data_path + "/json/parsed_papers.json"))
        self.incomplete = [x.strip() for x in open(data_path + "/txt/incomplete_papers.txt").readlines() if x != "\n"]
        self.papers = {}
        # for k,p in papers_dict.items():
        #     self.papers[k] = Paper(**p)
        self.author_papers = json.load(open(data_path + "/json/author_papers.json"))
        self.log_path = os.getcwd() + '/tests/authorDisambiguationTests/logs/'
        org_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                      open(data_path + "/txt/org_corpus.txt").readlines()]
        department_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                             open(data_path + "/txt/department_corpus.txt").readlines()]
        self.incomplete = [x.strip() for x in open(data_path + "/txt/incomplete_papers.txt").readlines()]
        self.compare_authors_args = {
            "company_corpus": org_corpus,
            "department_corpus": department_corpus,
            "threshold": .4,
            "str_algorithm": ["jaro", "similarity"]
        }
        self.id_to_name = json.load(open(data_path + "/json/id_to_name.json"))

    def compareInfoDict(self, actual, expected):
        self.assertTrue("name" in actual)
        self.assertEqual(actual["name"], cleanName(expected["name"]))
        self.assertTrue("co_authors_name" in actual)
        expected_names = [cleanName(x) for x in expected["co_authors_name"]]
        for n in actual["co_authors_name"]:
            self.assertTrue(n in expected_names)
        self.assertTrue("aff_name" in actual)
        if not expected["aff_name"]:
            self.assertEqual(actual["aff_name"], expected["aff_name"])
        else:
            self.assertEqual(actual["aff_name"], cleanName(expected["aff_name"]))
        self.assertTrue("address" in actual)
        try:
            self.assertDictEqual(actual["address"], expected["address"])
        except Exception as e:
            print(actual["address"])
            print(expected["address"])
            raise e
        self.assertTrue("title" in actual)
        self.assertEqual(actual["title"], cleanName(expected["title"]))
        list_keys = ["title_tokenized", "co_authors_id", "department", "co_authors_email",
                     "co_authors_aff", "co_authors_aff_type", "citations", "citations_tokenized", "sections",
                     "sections_tokenized"]
        str_keys = ["aff_type", "email_user", "email_domain"]
        for k in list_keys:
            try:
                self.assertTrue(k in actual)
            except Exception as e:
                print(k)
                raise e
            for i in actual[k]:
                try:
                    self.assertTrue(i in expected[k])
                except Exception as e:
                    print(actual["name"])
                    print(expected["name"])
                    print(i)
                    print(expected[k])
                    raise e
        for k in str_keys:
            self.assertTrue(k in actual)
            self.assertEqual(actual[k], expected[k])

    def compareList(self, a, b):
        missing = deepcopy(b)
        extra = []
        for i in a:
            if i in missing:
                missing.remove(i)
            else:
                extra.append(i)
        self.assertEqual([], missing)
        self.assertEqual([], extra)

    @ignore_warnings
    def test__findData(self):
        print("INFO: Running _findData tests")
        log_path = self.log_path + 'find_data.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(DEBUG_MODE=True, log_path=log_path)
        res, status, error_msg = author_processor._findData("blah.json")
        self.assertIsNone(res)
        self.assertEqual(-1, status)
        self.assertEqual("blah.json not found in any subdirectory", error_msg)
        res, status, error_msg = author_processor._findData("parsed_papers.json")
        self.assertEqual(0, status)
        self.assertEqual("", error_msg)

    def test__getAuthorInfos(self):
        print("INFO: Running _getAuthorInfos tests")
        log_path = self.log_path + 'get_author_info.log'
        with open(log_path, 'w'):
            pass
        test_auths = [
            "yang-liu-ict",
            "luyang-liu",
            "bob-newman",
            "yang-liu"
        ]
        test_papers = {
            "D17-1207": self.test_papers["D17-1207"],
            "C18-1172": self.test_papers["C18-1172"]
        }
        test_author_papers = {
            "yang-liu-ict": [
                "D17-1207"
            ],
            "luyang-liu": [
                "C18-1172"
            ],
            "bob-newman": [
                "A0-0000"
            ]
        }
        author_processor = AuthorDisambiguation(papers=test_papers, author_papers=test_author_papers,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                file_log_level=logging.WARNING)
        res, error_auth, error_paper = author_processor._getAuthorInfos(test_auths)
        self.assertEqual(1, error_auth)
        self.assertEqual(1, error_paper)
        for i, v in res.items():
            if i == "D17-1207 yang-liu-ict":
                self.compareInfoDict(v, getAuthorInfo([self.test_papers["D17-1207"], "yang-liu-ict"])[1])
            elif i == "C18-1172 luyang-liu":
                self.compareInfoDict(v, getAuthorInfo([self.test_papers["C18-1172"], "luyang-liu"])[1])

    @ignore_warnings
    def test__getSimilarAuthors(self):
        print("INFO: Running _getSimilarAuthors tests")
        log_path = self.log_path + 'get_similar_authors.log'
        with open(log_path, 'w'):
            pass
        expected = [
            "yang-liu-edinburgh",
            "yang-liu-ict",
            "yang-liu",
            "yang-liu-icsi",
            "yang-li",
        ]
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95)
        res = author_processor._getSimilarAuthors("yang-liu", "yang liu")
        self.compareList(expected, res)
        expected_2 = [
            "yang-liu-georgetown",
            "yang-liu-edinburgh",
            "yang-liu-ict",
            "yang-liu",
            "yang-liu-icsi",
            "yang-li"
        ]
        author_processor.sim_overrides = True
        res = author_processor._getSimilarAuthors("yang-liu", "yang liu")
        self.compareList(expected_2, res)
        print(author_processor._getSimilarAuthors("eugenio-martinez-camara1", "Eugenio Martinez Camara".lower()))

    @ignore_warnings
    def test__makePairs(self):
        print("INFO: Running _makePairs tests")
        log_path = self.log_path + 'make_pairs.log'
        with open(log_path, 'w'):
            pass
        test_auths = [
            ["A1-1000 yang-liu", 1],
            ["A1-1001 yang-liu", 1],
            ["A1-1002 yang-liu", 1],
            ["A1-1003 yang-liu", 1],
            ["A1-1004 yang-liu", 1],
            ["A1-1005 yang-liu", 1],
        ]
        test_auth = ["A1-1002 yang-liu", 1]
        expected_out = [
            ["A1-1002 yang-liu A1-1000 yang-liu", 1, 1],
            ["A1-1002 yang-liu A1-1001 yang-liu", 1, 1],
            ["A1-1002 yang-liu A1-1003 yang-liu", 1, 1],
            ["A1-1002 yang-liu A1-1004 yang-liu", 1, 1],
            ["A1-1002 yang-liu A1-1005 yang-liu", 1, 1],
        ]
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95)
        res, excluded = author_processor._makePairs(test_auth, test_auths)
        self.assertEqual([["A1-1002 yang-liu", 1]], excluded)
        self.compareList(expected_out, res)

    @ignore_warnings
    def test__determineCorrectAuthor(self):
        test_1 = {
            "a": [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            "b": [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            "c": [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            "d": [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            "e": [1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            "f": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "g": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        }
        test_2 = {
            "a": [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            "b": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "c": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "d": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "e": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "f": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "g": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        test_3 = {
            "a": [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            "b": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "c": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "d": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "e": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "f": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "g": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        test_4 = {
            "a": [1, 1, 1, 1, 1, 0],
            "b": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],

        }
        print("INFO: Running determineCorrect tests")
        log_path = self.log_path + 'determine_correct.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95)
        res, above = author_processor._determineCorrectAuthor(test_1)
        self.assertEqual("a", res)
        self.assertEqual(1, above)
        res, above = author_processor._determineCorrectAuthor(test_2)
        self.assertEqual("b", res)
        self.assertEqual(2, above)
        res, above = author_processor._determineCorrectAuthor(test_3)
        self.assertIsNone(res)
        self.assertEqual(0, above)
        res, above = author_processor._determineCorrectAuthor(test_4)
        self.assertEqual("b", res)
        self.assertEqual(2, above)

    @ignore_warnings
    def test_checkCallErrors(self):
        print("INFO: Running checkCallError tests")
        log_path = self.log_path + 'call_error_check.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, allow_authors_not_in_override=False)
        test_1 = [["abcsasd-adad"], {}]
        test_2 = [["yang-liu"], {
            "abc-de": ["yang-liu-ict"]
        }]
        test_3 = [["yang-liu"], {
            "yang-liu": "yang-liu-ict"
        }]
        test_4 = [["yang-liu"], {
            "yang-liu": "yang-liu-ict"
        }]
        test_5 = [["yang-liu"], {
            "yang-liu": ["yang-liu-ict", "yang-liu"]
        }]
        test_6 = [["yang-liu"], {
            "yang-liu": ["yang-liu-ict", "no-one-should-ever-have-this-name"]
        }]

        with self.assertRaises(ValueError):
            author_processor._errorCheckCallArgs(test_2[0], test_2[1])
            author_processor._errorCheckCallArgs(test_3[0], test_3[1])
            author_processor._errorCheckCallArgs(test_4[0], test_4[1])
            author_processor._errorCheckCallArgs(test_5[0], test_5[1])

        with self.assertRaises(KeyError):
            author_processor._errorCheckCallArgs(test_1[0], test_1[1])
            author_processor._errorCheckCallArgs(test_6[0], test_6[1])

        should_work_targets = ["luyang-liu", "yang-liu"]
        should_work_override = {
            "yang-liu": [
                "yang-liu-ict"
            ]
        }
        a, b = author_processor._errorCheckCallArgs(should_work_targets, should_work_override)
        self.assertEqual(["yang-liu"], a)
        self.assertEqual(["luyang-liu"], b)

    @ignore_warnings
    def test_compareAuthors(self):
        test_target = ["D17-1207", "yang-liu-ict"]
        test = [
            ["C10-2059", "yajuan-lu"],
            ["P16-1159", "yong-cheng"],
            ["P09-2066", "yang-liu-icsi"]
        ]
        info_dict = {
            test_target[0] + " " + test_target[1]: getAuthorInfo([self.test_papers[test_target[0]], test_target[1]])[1]
        }
        pairs = []
        for p, n in test:
            info_dict[p + " " + n] = getAuthorInfo([self.test_papers[p], n])[1]
            pairs.append([" ".join([*test_target, p, n]), info_dict[" ".join(test_target)], info_dict[p + " " + n]])

        print("INFO: Running compareAuthors tests")
        log_path = self.log_path + 'compare_authors.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, allow_authors_not_in_override=False)
        comparator = CompareAuthors(**self.compare_authors_args)
        key, res = author_processor._compareAuthors([comparator, " ".join(test_target), pairs])
        self.assertEqual(" ".join(test_target), key)
        self.assertNotEqual(0, len(res))
        for k, info in info_dict.items():
            if k == " ".join(test_target):
                continue
            k_id = k.split()[1]
            self.assertTrue(k_id in res)
            self.assertEqual(1, len(res[k_id]))
            expected = comparator([" ".join([*test_target, k]), 0, info_dict[" ".join(test_target)], info])[-1]
            np.testing.assert_array_equal(expected, res[k_id][0])

    @ignore_warnings
    def test_makeAmbiguousAuthors(self):
        print("INFO: Running makeAmbiguousAuthor tests")
        log_path = self.log_path + 'make_ambiguous_author.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, sim_overrides=True)
        test_override = {
            "luyang-liu": ["bo-li"]
        }
        test_has_authors = ["luyang-liu"]
        test_no_authors = ["eugenio-martinez-camara", "yang-liu"]
        expected_authors_to_get = ["bo-li",
                                   "yang-liu-georgetown",
                                   "yang-liu-edinburgh",
                                   "yang-liu-ict",
                                   "yang-liu-icsi",
                                   "yang-li"]
        expected_excluded = ["eugenio-martinez-camara"]
        expected_names = {
            "yang-liu": "yang liu",
            "luyang-liu": "luyang liu"
        }
        expected_author_papers = {
            "luyang-liu": [
                "C18-1172"
            ],
            "yang-liu": [
                "K18-1018",
                "I13-1154",
                "C12-2073"
            ],
            "eugenio-martinez-camara": [
                "W17-6927",
                "W17-0908",
                "W18-6227"
            ]
        }
        tmp_authors = []
        for i in expected_authors_to_get:
            if "yang-li" not in i:
                continue
            tmp_authors.extend([(p, i) for p in self.author_papers[i]])
        expected_check_authors = {
            "luyang-liu": [
                ("D18-1212", "bo-li"),
                ("C18-1025", "bo-li"),
                ("P19-1130", "bo-li")
            ],
            "yang-liu": tmp_authors

        }
        res = author_processor._makeAmbiguousAuthors(test_has_authors, test_no_authors, test_override)
        ambiguous_author_papers, ambiguous_author_names, check_author_keys, authors_get_info, excluded = res
        self.assertEqual(expected_excluded, excluded)
        for a in ambiguous_author_papers.keys():
            if a not in expected_author_papers:
                print(a)
                self.assertTrue(a in expected_author_papers)
            self.compareList(ambiguous_author_papers[a], expected_author_papers[a])
            self.assertTrue(a not in author_processor.author_papers)
        for k, n in ambiguous_author_names.items():
            self.assertTrue(k in expected_names)
            self.assertEqual(expected_names[k], n)

        self.compareList(authors_get_info, expected_authors_to_get)

        for k, i in check_author_keys.items():
            if k not in expected_check_authors:
                self.assertTrue(k in expected_check_authors)
            self.compareList(i, expected_check_authors[k])

    def test_makeAmbiguousPairs(self):
        print("INFO: Running makeAmbiguousPairs tests")
        log_path = self.log_path + 'makeAmbiguousPairs.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, sim_overrides=True)

        ambiguous_papers = {
            "yang-liu-georgetown": [
                "W19-2708",
                "W19-2710",
                "W19-2717"
            ]
        }
        check_authors = {
            "yang-liu-georgetown": {
                ("W19-2708", "amir-zeldes"),
                ("Q18-1005", "yang-liu-edinburgh"),
                ("N19-1173", "yang-liu-edinburgh"),
                ("P15-2047", "yang-liu-edinburgh")
            },
        }
        authors_to_get = ["amir-zeldes", "yang-liu-edinburgh"]

        results, excluded = author_processor._makeAmbiguousPairs(ambiguous_papers, check_authors, authors_to_get)
        expected_excluded = {
            'W19-2708 yang-liu-georgetown': ['W19-2708 amir-zeldes']
        }
        for k in expected_excluded.keys():
            self.assertEqual(expected_excluded[k], excluded[k])
        expected_results = {
            "W19-2708 yang-liu-georgetown": [
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ],
            "W19-2710 yang-liu-georgetown": [
                "W19-2708 amir-zeldes",
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh",
            ],
            "W19-2717 yang-liu-georgetown": [
                "W19-2708 amir-zeldes",
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ]
        }
        for k, info in results.items():
            if k not in expected_results:
                print(k)
                self.fail()
            results_pair_keys = [x[0] for x in info]
            expected_pair_keys = [" ".join([k, x]) for x in expected_results[k]]
            self.compareList(results_pair_keys, expected_pair_keys)

    def test_removeKnownDifferent(self):
        print("INFO: Running removeKnownDifferent tests")
        log_path = self.log_path + 'remove_known_different.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, sim_overrides=True)

        tmp_pairs = {
            "W19-2708 yang-liu-georgetown": [
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ],
            "W19-2710 yang-liu-georgetown": [
                "W19-2708 amir-zeldes",
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh",
            ],
            "W19-2717 yang-liu-georgetown": [
                "W19-2708 amir-zeldes",
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ],
            "Q18-1005 yang-liu-edinburgh": [
                "W19-2708 amir-zeldes",
            ]
        }
        test_pairs = {}
        for k, info in tmp_pairs.items():
            test_pairs[k] = [[" ".join([k, x]), 1] for x in info]
        test_excluded = {
            'W19-2708 yang-liu-georgetown': ['W19-2708 amir-zeldes'],
            "W19-2710 yang-liu-georgetown": []
        }
        expected_different = {
            "yang-liu-georgetown": ["amir-zeldes"]
        }
        expected_pairs = {
            "W19-2708 yang-liu-georgetown": [
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ],
            "W19-2710 yang-liu-georgetown": [
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh",
            ],
            "W19-2717 yang-liu-georgetown": [
                "Q18-1005 yang-liu-edinburgh",
                "N19-1173 yang-liu-edinburgh",
                "P15-2047 yang-liu-edinburgh"
            ],
            "Q18-1005 yang-liu-edinburgh": [
                "W19-2708 amir-zeldes",
            ]
        }
        fixed, diff = author_processor._removeKnownDifferent(test_pairs, test_excluded)
        self.assertDictEqual(expected_different, diff)
        for k, info in fixed.items():
            if k not in expected_pairs:
                print(k)
                self.fail()
            results_pair_keys = [x[0] for x in info]
            expected_pair_keys = [" ".join([k, x]) for x in expected_pairs[k]]
            self.compareList(results_pair_keys, expected_pair_keys)

    @ignore_warnings
    def test_consolidateResults(self):
        print("INFO: Running consolidateResults tests")
        log_path = self.log_path + 'consolidate_results.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, sim_overrides=True)
        expected_compare_array = np.array([1 for x in range(24)])

        test_results = {
            "W19-2708 yang-liu-georgetown": {
                "yang-liu-edinburgh": [[1 for x in range(24)] for x in range(3)]
            },
            "W19-2710 yang-liu-georgetown": {
                "yang-liu-edinburgh": [[1 for x in range(24)] for x in range(2)]
            },
            "W19-2717 yang-liu-georgetown": {
                "yang-liu-edinburgh": [[1 for x in range(24)] for x in range(1)]
            },
            "Q18-1005 yang-liu-edinburgh": {
                "amir-zeldes": [[1 for x in range(24)] for x in range(1)]
            }
        }
        res = author_processor._consolidateResults(test_results)
        self.assertEqual(2, len(res))
        self.assertEqual(["yang-liu-georgetown", "yang-liu-edinburgh"], list(res.keys()))
        for k, items in res.items():
            self.assertEqual(1, len(items))
            for a, r in items.items():
                if k == "yang-liu-georgetown":
                    self.assertEqual(6, r.shape[0])
                elif k == "yang-liu-edinburgh":
                    self.assertEqual(1, r.shape[0])
                for i in range(r.shape[0]):
                    np.testing.assert_array_equal(r[i], expected_compare_array)

    @ignore_warnings
    def test_makePredictions(self):
        print("INFO: Running makePredictions tests")
        log_path = self.log_path + 'make_predictions.log'
        with open(log_path, 'w'):
            pass
        author_processor = AuthorDisambiguation(papers=self.test_papers, id_to_name=self.id_to_name,
                                                compare_args=self.compare_authors_args, log_path=log_path,
                                                name_similarity_cutoff=.95, sim_overrides=True,model_path=os.getcwd(),model_name="SoftVoting")

        test_target = ["D17-1207", "yang-liu-ict"]
        test = [
            ["C10-2059", "yajuan-lu"],
            ["P16-1159", "yong-cheng"],
            ["P09-2066", "yang-liu-icsi"],
            ["D14-1076","yang-liu-icsi"],
            ["D15-1210","yang-liu-ict"],
            ["P16-1159","yang-liu-ict"]
        ]
        info_dict = {
            test_target[0] + " " + test_target[1]: getAuthorInfo([self.test_papers[test_target[0]], test_target[1]])[1]
        }
        pairs = []
        for p, n in test:
            info_dict[p + " " + n] = getAuthorInfo([self.test_papers[p], n])[1]
            pairs.append([" ".join([*test_target, p, n]), info_dict[" ".join(test_target)], info_dict[p + " " + n]])
        comparator = CompareAuthors(**self.compare_authors_args)
        key, res = author_processor._compareAuthors([comparator, " ".join(test_target), pairs])
        test_compare_results ={key: res}
        consolidated = author_processor._consolidateResults(test_compare_results)
        predictions,probabilities = author_processor._makePredictions(consolidated)
        for k,info in predictions.items():
            self.assertTrue(k in probabilities)
            for a, predict in info.keys():
                # self.assertTrue( a in probabilities[k])
                if a =="yang-liu-icsi" or a == "yang-liu-ict":
                    self.assertEqual(2,len(predict))
                    # self.assertEqual(2, len(probabilities[k][a]))
                else:
                    self.assertEqual(1,len(predict))
                    # self.assertEqual(1,len(probabilities[k][a]))