from unittest import TestCase
import os
import json
import logging
from copy import deepcopy
from src.utility_functions import *
from src.paper import Paper
import warnings
import sys
from src.create_training_data import CreateTrainingData
from src.paper import Paper
import time
import numpy as np
from src.utility_functions import cleanName
from src.compare_authors import compareAuthors
from textdistance import JaroWinkler
from tqdm import tqdm

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestCreateAuthorPairs(TestCase):
    @ignore_warnings
    def setUp(self):
        self.config = json.load(open("/home/gabe/Desktop/research-main/config.json"))
        data_path = "/home/gabe/Desktop/research-main/data"
        papers_dict = json.load(open(data_path + "/json/parsed_papers.json"))
        self.test_auth_info = json.load(open("/home/gabe/Desktop/research-main/tests/createPairTests/test_papers.json"))
        self.incomplete = [x.strip() for x in open(data_path + "/txt/incomplete_papers.txt").readlines()]
        self.test_papers = {
            "N12-1057": {
                "owen-rambow": "Owen Rambow",
                "mona-diab": "Mona Diab",
                "vinodkumar-prabhakaran": "Vinodkumar Prabhakaran"
            },
            "N19-1050": {
                "shima-asaadi": "Shima Asaadi",
                "saif-mohammad": "Saif Mohammad",
                "svetlana-kiritchenko": "Svetlana Kiritchenko"
            },
            "C16-1050": {
                "elaheh-shafieibavani": "Elaheh ShafieiBavani",
                "mohammad-ebrahimi": "Mohammad Ebrahimi",
                "raymond-wong": "Raymond Wong",
                "fang-chen": "Fang Chen"
            },
            "S19-2016": {
                "tobias-putz": "Tobias P\u00fctz",
                "kevin-glocker": "Kevin Glocker"
            },
            "P19-1642": {
                "iacer-calixto": "Iacer Calixto",
                "miguel-rios": "Miguel Rios",
                "wilker-aziz": "Wilker Aziz"
            },
            "W19-4022": {
                "jungyeul-park": "Jungyeul Park",
                "francis-tyers": "Francis Tyers"
            },
            "Q19-1001": {
                "dan-roth": "Dan Roth",
                "alla-rozovskaya": "Alla Rozovskaya"
            },
            "P15-1150": {
                "christopher-d-manning": "Christopher D. Manning",
                "kai-sheng-tai": "Kai Sheng Tai",
                "richard-socher": "Richard Socher"
            },
        }
        self.test_keys = []
        for k, info in self.test_papers.items():
            for a in info.keys():
                self.test_keys.append(k + " " + a)

        self.papers = {}
        self.short_papers = {}
        for k, info in papers_dict.items():
            if k in self.test_papers or k in self.incomplete:
                self.short_papers[k] = Paper(**papers_dict[k])
            self.papers[k] = Paper(**papers_dict[k])
            for a, aff_info in info["affiliations"].items():
                if "type" in aff_info and len(aff_info["type"]) > 2:
                    print(k)
                    print(info)
                    break

    def checkIgnored(self, ignored):
        for i in self.incomplete:
            self.assertTrue(i in ignored)

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
        self.assertDictEqual(actual["address"], expected["address"])
        self.assertTrue("title" in actual)
        self.assertEqual(actual["title"], cleanName(expected["title"]))
        list_keys = ["title_tokenized", "title_pos", "co_authors_id", "department"]
        str_keys = ["aff_type", "email_user", "email_domain"]
        for k in list_keys:
            self.assertTrue(k in actual)
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

    def test__getAlgo(self):
        pass

    def test__createGetAuthorInfoArgs(self):
        pair_creator = CreateTrainingData(self.short_papers, self.incomplete)
        total, results, ignored = pair_creator._createGetAuthorInfoArgs()
        self.checkIgnored(ignored)
        for p, a in results:
            tmp_key = p.pid + " " + a
            self.assertTrue(tmp_key in self.test_keys)
        self.assertEqual(total, len(self.test_keys))

    def test__getAuthorInfo(self):
        pair_creator = CreateTrainingData(self.short_papers, self.incomplete)
        total, tasks, ignored = pair_creator._createGetAuthorInfoArgs()
        expected_total = 0
        for k, info in self.short_papers.items():
            if k in self.incomplete:
                continue
            for a in info.affiliations.keys():
                expected_total += 1
        self.checkIgnored(ignored)
        results = []
        self.assertEqual(len(tasks), expected_total)
        for i in tasks:
            pair_key, res = pair_creator._getAuthorInfo(i)
            if pair_key in self.test_auth_info:
                self.compareInfoDict(res, self.test_auth_info[pair_key])
            results.append((pair_key, res))
        self.assertEqual(len(results), expected_total)

    def test_runTimes(self):
        expected_total = 0
        for k, info in self.papers.items():
            if k in self.incomplete:
                continue
            for a in info.affiliations.keys():
                expected_total += 1
        run_times = []
        pair_creator = CreateTrainingData(self.papers, self.incomplete)
        total, tasks, ignored = pair_creator._createGetAuthorInfoArgs()
        self.assertEqual(total, expected_total)
        self.checkIgnored(ignored)
        results = []
        for i in tasks:
            t0 = time.time()
            res = pair_creator._getAuthorInfo(i)
            t1 = time.time()
            results.append(res)
            run_times.append(t1 - t0)
        run_times = np.asarray(run_times)
        print("Average runtime to get info {:.3E}s".format(np.mean(run_times)))
        self.assertEqual(len(results), expected_total)
        algorithm = JaroWinkler().similarity
        run_times = []
        comp_results = []
        tstart = time.time()
        for i,x in tqdm(enumerate(results[:500]),file=sys.stdout):
            for j,y in enumerate(results[i+1:500]):
                t0 = time.time()
                _,_,res = compareAuthors(["",0,x[1],y[1],algorithm])
                t1 = time.time()
                comp_results.append(res)
                run_times.append(t1 - t0)
        tend = time.time()
        run_times = np.asarray(run_times)
        print("Average runtime to compare authors {:.3E}s".format(np.mean(run_times)))
        print("Total time to compare authors {:.3f}s".format(np.mean(tend-tstart)))
        self.assertEqual(len(comp_results),124750)
    def test_compareAuthors(self):
        algorithm = JaroWinkler().similarity
        # [
        #     first_name_score,
        #     middle_name_score,
        #     last_name_score,
        #     initials_score,
        #     org_name_score,
        #     org_type_score,
        #     email_domain_score,
        #     email_user_score,
        #     co_auth_score,
        #     department_score,
        #     same_title_words,
        #     "postCode",
        #     "settlement",
        #     "country"
        #  ]
        test_1 = [
            algorithm("Iacer","Elaheh"),
            1.0,
            algorithm("Calixto","ShafieiBavani"),
            0.0,
            algorithm("ILLC The University of Amsterdam","University of New South Wales"),
            1.0,
            algorithm("uva.nl","cse.unsw.edu.au"),
            algorithm("iacer.calixto","elahehs"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        test_2 = [
            algorithm("Iacer", "Elaheh"),
            1.0,
            algorithm("Calixto", "ShafieiBavani"),
            0.0,
            algorithm("ILLC The University of Amsterdam", "University of New South Wales"),
            1.0,
            algorithm("uva.nl", "cse.unsw.edu.au"),
            algorithm("iacer.calixto", "elahehs"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        a = self.test_auth_info["P19-1642 iacer-calixto"]
        b = self.test_auth_info["C16-1050 elaheh-shafieibavani"]
        args = [
            ["",0,a, b, algorithm]
        ]
        run_times =[]
        for arg in args:
            k,t,res = compareAuthors(arg)
            self.assertEqual(k,arg[0])
            self.assertEqual(t,arg[1])
            self.assertEqual(res.tolist(),test_1)
        # print("Average runtime to compare authors {:.3E}s".format(np.mean(run_times)))