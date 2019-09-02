from unittest import TestCase
import os
import json
import logging
from copy import deepcopy
from src.utility_functions import *
from src.paper import Paper
from py_stringmatching.similarity_measure import soft_tfidf
import warnings
import sys
from src.create_training_data import CreateTrainingData, getAuthorInfo
from src.paper import Paper
import time
import numpy as np
from src.utility_functions import cleanName
from textdistance import JaroWinkler
from tqdm import tqdm
import multiprocessing as mp


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestCreateTrainingData(TestCase):
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
            'P17-1139': {
                'yang-liu-ict': 'Yang Liu',
                'maosong-sun': 'Maosong Sun',
                'jiacheng-zhang': 'Jiacheng Zhang',
                'huanbo-luan': 'Huanbo Luan',
                'jingfang-xu': 'Jingfang Xu'
            },
            'C10-2136': {
                'yang-liu-ict': 'Yang Liu',
                'yajuan-lu': 'Yajuan Lv',
                'qun-liu': 'Qun Liu',
                'jinsong-su': 'Jinsong Su',
                'haitao-mi': 'Haitao Mi',
                'hongmei-zhao': 'Hongmei Zhao'
            },
            'D18-1041': {
                'yang-liu-ict': 'Yang Liu',
                'jinsong-su': 'Jinsong Su',
                'jiali-zeng': 'Jiali Zeng',
                'huating-wen': 'Huating Wen',
                'jun-xie': 'Jun Xie',
                'yongjing-yin': 'Yongjing Yin',
                'jianqiang-zhao': 'Jianqiang Zhao'
            },
            'Q18-1029': {
                'yang-liu-ict': 'Yang Liu',
                'zhaopeng-tu': 'Zhaopeng Tu',
                'shuming-shi': 'Shuming Shi',
                'tong-zhang': 'Tong Zhang'
            },
            'P17-1176': {
                'yang-liu-ict': 'Yang Liu',
                'victor-ok-li': 'Victor O.K. Li',
                'yun-chen': 'Yun Chen',
                'yong-cheng': 'Yong Cheng'
            },
            'P09-1065': {
                'yang-liu-ict': 'Yang Liu',
                'qun-liu': 'Qun Liu',
                'haitao-mi': 'Haitao Mi',
                'yang-feng': 'Yang Feng'
            },
            'P13-1084': {
                'yang-liu-ict': 'Yang Liu',
                'jun-zhao': 'Jun Zhao',
                'guangyou-zhou': 'Guangyou Zhou',
                'shizhu-he': 'Shizhu He',
                'fang-liu': 'Fang Liu'
            }
        }
        self.test_keys = []
        for k, info in self.test_papers.items():
            for a in info.keys():
                if a == "yang-feng":
                    continue
                self.test_keys.append(k + " " + a)

        self.papers = {}
        self.short_papers = {}
        for k, info in papers_dict.items():
            if k in self.test_papers or k in self.incomplete:
                self.short_papers[k] = Paper(**papers_dict[k])
            self.papers[k] = Paper(**papers_dict[k])
            for a, aff_info in info["affiliations"].items():
                if a == "yang-feng":
                    continue
                if "type" in aff_info and len(aff_info["type"]) > 2:
                    print(k)
                    print(info)
                    break

        self.default_args = dict(author_cutoff=0,drop_null_authors=False)
        self.log_path = os.getcwd()+'/createPairTests/logs/'

    def checkIgnored(self, ignored):
        for i in self.incomplete:
            self.assertTrue(i in ignored)

    def checkNotInCombinations(self, combos, _id):
        for k, _, _ in combos:
            self.assertTrue(_id not in k)

    def checkNoSameCombinations(self, combos):
        for k, _, _ in combos:
            p1, id1, p2, id2 = k.split()
            self.assertNotEqual(p1 + " " + id1, p2 + " " + id2)

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
                     "co_authors_aff","co_authors_aff_type","citations","citations_tokenized","sections","sections_tokenized"]
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

    def test__populateConstants(self):
        log_path = self.log_path+'populate_constants.log'
        with open(log_path, 'w'):
            pass
        pair_creator = CreateTrainingData(self.short_papers, self.incomplete, exclude=["yang-feng"],
                                          **self.default_args,log_path=log_path)
        total, results, ignored, excluded = pair_creator._populateConstants()
        self.checkIgnored(ignored)
        for p, a in results:
            tmp_key = p.pid + " " + a
            self.assertTrue(tmp_key in self.test_keys)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded, [("P09-1065", "yang-feng")])
        self.assertEqual(total, len(self.test_keys))
        test_auths = ["yang-liu-ict", "qun-liu", "jinsong-su", "iacer-calixto"]
        test_auth_counts = {x: [] for x in test_auths}
        for k, a in self.test_papers.items():
            for author in test_auths:
                if author in a:
                    test_auth_counts[author].append(k)

        for a in test_auth_counts:
            self.assertAlmostEqual(len(pair_creator.valid_author_papers[a]), len(test_auth_counts[a]))
            for p in test_auth_counts[a]:
                self.assertTrue(p in pair_creator.valid_author_papers[a])

    def test__getAuthorInfo(self):
        log_path = self.log_path + 'get_author_info.log'
        with open(log_path, 'w'):
            pass
        pair_creator = CreateTrainingData(self.short_papers, self.incomplete, exclude=["yang-feng"],
                                          **self.default_args,log_path=log_path)
        total, tasks, ignored, excluded = pair_creator._populateConstants()
        expected_total = 0
        for k, info in self.short_papers.items():
            if k in self.incomplete:
                continue
            for a in info.affiliations.keys():
                if a == "yang-feng":
                    continue
                expected_total += 1
        self.checkIgnored(ignored)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded, [("P09-1065", "yang-feng")])
        results = []
        self.assertEqual(len(tasks), expected_total)
        for i in tasks:
            pair_key, res = getAuthorInfo(i)
            if pair_key in self.test_auth_info:
                self.compareInfoDict(res, self.test_auth_info[pair_key])
            results.append((pair_key, res))
        self.assertEqual(len(results), expected_total)

    def test_createPairDict(self):
        log_path = self.log_path + 'create_pair_dict.log'
        with open(log_path, 'w'):
            pass
        test_pairs = [
            'N12-1057 john-wilkins',
            'N12-1057 mona-diab',
            'N12-1057 john-smith',
            'N19-1050 shima-asaadi',
            'N19-1050 saif-mohammad',
            'N19-1050 svetlana-kiritchenko',
            'C16-1050 john-doe',
            'C16-1050 josh-way',
            'C16-1050 jeff-wilkins',
            'S19-2016 sasha-mohammad',
            'S19-2016 john-wilson',
            'P19-1642 miguel-rios',
        ]
        test_1 = {
            "j": 6,
            "m": 2,
            "s": 4
        }
        test_2 = {
            "jo": 5,
            "sa": 2
        }

        pair_creator = CreateTrainingData(self.papers, self.incomplete, **self.default_args,log_path=log_path)
        res = pair_creator._createPairDict(test_pairs)
        self.assertTrue("j" in res)
        self.assertTrue("m" in res)
        self.assertTrue("s" in res)
        self.assertTrue("N12-1057 john-wilkins" in res['j'])
        for k in test_1.keys():
            self.assertEqual(len(res[k]), test_1[k])
        res = pair_creator._createPairDict(test_pairs, 2)
        for k in test_2.keys():
            self.assertTrue(k in res)
            self.assertEqual(len(res[k]), test_2[k])
        self.assertTrue("N12-1057 john-wilkins" in res['jo'])
        res = pair_creator._createPairDict(test_pairs, char_count=6, word_count=2)
        self.assertTrue("john w" in res)
        self.assertTrue('N12-1057 john-wilkins' in res["john w"])

    def test_makeCombinations(self):
        log_path = self.log_path + 'make_combinations.log'
        with open(log_path, 'w'):
            pass
        t1 = [
            ('N12-1057 person-a', 1),
            ('N12-1057 person-b', 2),
            ('N12-1058 person-a', 3),
            ('N12-1058 person-c', 4),
            ('N12-1058 person-d', 5),
            ('N12-1059 person-a', 6),
            ('N12-1059 person-c', 7),
            ('N12-1060 human-a', 8),
            ('N12-1061 human-b', 9),
        ]
        pair_a = ['N12-1057 person-a N12-1058 person-a', 'N12-1057 person-a', 'N12-1058 person-a']
        pair_b = ['N12-1057 person-a N12-1057 person-b', 'N12-1057 person-a', 'N12-1057 person-b']
        pair_c = ['N12-1057 person-a N12-1058 person-c', 'N12-1057 person-a', 'N12-1058 person-c']
        pair_d = ['N12-1057 person-a N12-1058 person-d', 'N12-1057 person-a', 'N12-1058 person-d']
        pair_e = ['N12-1057 person-a N12-1060 human-a', 'N12-1057 person-a', 'N12-1060 human-a']
        pair_f = ['N12-1060 human-a N12-1061 human-b', 'N12-1060 human-a', 'N12-1061 human-b']
        pair_g = ['N12-1058 person-c N12-1057 person-a', 'N12-1060 human-a', 'N12-1061 human-b']
        pair_creator = CreateTrainingData(self.papers, self.incomplete, name_similarity_cutoff=.8,
                                          cores=1, **self.default_args,log_path=log_path)
        algorithm = pair_creator.algorithm
        s, d = pair_creator._makeCombinations(t1)
        self.checkNotInCombinations(s, 'N12-1057 person-a N12-1060 human-a')
        self.checkNotInCombinations(d, 'N12-1057 person-a N12-1060 human-a')
        self.checkNoSameCombinations(s)
        self.checkNoSameCombinations(d)
        self.assertTrue(pair_a in s)
        self.assertTrue(pair_a not in d)
        self.assertTrue(pair_b not in s)
        self.assertTrue(pair_b not in d)
        self.assertTrue(pair_c not in s)
        self.assertTrue(pair_c in d)
        self.assertTrue(pair_d not in s)
        self.assertTrue(pair_d in d)
        self.assertTrue(pair_e not in s)
        self.assertTrue(pair_e not in d)
        self.assertTrue(pair_g not in s)
        self.assertTrue(pair_g not in d)
        s, d = pair_creator._makeCombinations(t1, use_cutoff=False)
        self.assertTrue(pair_e not in s)
        self.assertTrue(pair_e in d)
        s, d = pair_creator._makeCombinations(t1, special_cases=["N12-1060 human-a", "N12-1061 human-b"],
                                              use_cutoff=False)
        self.assertTrue(pair_f not in s)
        self.assertTrue(pair_f not in d)

    def test_getSpecialCases(self):
        log_path = self.log_path + 'get_special_cases.log'
        with open(log_path, 'w'):
            pass
        t1 = {
            "p": [
                'N12-1057 person-a',
                'N12-1057 person-b',
                'N12-1058 person-a',
                'N12-1058 person-c',
                'N12-1058 person-d',
                'N12-1059 person-a',
                'N12-1059 person-c',
            ]
        }
        t2 = {
            "person a": [
                'N12-1057 person-abc',
                'N12-1058 person-abc',
                'N12-1058 person-ade',
                'N12-1058 person-acd',
                'N12-1059 person-abe',
                'N12-1059 person-acd'
            ]
        }
        pair_creator = CreateTrainingData(self.papers, self.incomplete, special_keys=["person-a"],
                                          cores=1, **self.default_args,log_path=log_path)
        special_cases = pair_creator._getSpecialCases(t1)
        self.assertEqual(len(special_cases), 1)
        self.assertTrue("p" in special_cases)
        self.assertEqual(special_cases["p"], [
            'N12-1057 person-a',
            'N12-1058 person-a',
            'N12-1059 person-a',
        ])
        pair_creator = CreateTrainingData(self.papers, self.incomplete, special_keys=["person-ab"], separate_chars=8,
                                          separate_words=2, cores=1, **self.default_args,log_path=log_path)
        special_cases = pair_creator._getSpecialCases(t2)
        self.assertEqual(len(special_cases), 1)
        self.assertTrue("person a" in special_cases)
        self.assertEqual(special_cases["person a"], [
            'N12-1057 person-abc',
            'N12-1058 person-abc',
            'N12-1059 person-abe',
        ])

    def test_prepareData(self):
        log_path = self.log_path + 'prepare_data.log'
        with open(log_path, 'w'):
            pass
        t1 = {
            'N12-1057 student-a1': 1,
            'N12-1057 professor-a': 2,
            'N12-1058 student-a2': 3,
            'N12-1058 professor-a': 5,
            'N12-1059 student-b': 6,
            'N12-1059 professor-b': 7,
            'N12-1060 professor-b': 8,
            'N12-1060 professor-a': 9,
            'N12-1060 student-b': 10,
            'N12-1060 student-a1': 4,
        }
        pair_creator = CreateTrainingData(self.papers, self.incomplete, name_similarity_cutoff=.8, special_keys=[
            "student-a"], cores=1, **self.default_args,log_path=log_path)
        algorithm = pair_creator.algorithm
        separated = pair_creator._createPairDict(list(t1.keys()))
        self.assertEqual(len(separated["p"]), 5)
        self.assertEqual(len(separated['s']), 5)
        same, diff, special_same, special_diff = pair_creator._prepareData(separated, t1, algorithm)
        self.assertEqual(len(same), 5)
        self.assertEqual(len(diff), 10)
        self.assertEqual(len(special_diff), 2)
        self.assertEqual(len(special_same), 1)

    def test_selectPairs(self):
        log_path = self.log_path + 'select_pairs.log'
        with open(log_path, 'w'):
            pass
        test_diff = [x for x in range(100)]
        test_same = [x for x in range(100)]
        pair_creator = CreateTrainingData(self.papers, self.incomplete, **self.default_args,log_path=log_path)
        same, diff = pair_creator._selectPairsToUse(test_same[:25], test_diff)
        self.assertEqual(len(same), 25)
        self.assertEqual(len(diff), 50)
        same, diff = pair_creator._selectPairsToUse(test_same, test_diff[:25])
        self.assertEqual(len(same), 50)
        self.assertEqual(len(diff), 25)
        pair_creator.dif_same_ratio = 1.5
        same, diff = pair_creator._selectPairsToUse(test_same, test_diff[:20])
        self.assertEqual(len(same), 30)
        self.assertEqual(len(diff), 20)
        same, diff = pair_creator._selectPairsToUse(test_same[:20], test_diff)
        self.assertEqual(len(same), 20)
        self.assertEqual(len(diff), 30)


"""
    def test_runTimes(self):
        expected_total = 0
        for k, info in self.papers.items():
            if k in self.incomplete:
                continue
            for a in info.affiliations.keys():
                expected_total += 1
        run_times = []
        pair_creator = CreateTrainingData(self.papers, self.incomplete)
        total, tasks, ignored = pair_creator._populateConstants()
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
        pbar = tqdm(total=len(results[:500]), file=sys.stdout)
        for i, x in enumerate(results[:500]):
            for j, y in enumerate(results[i + 1:500]):
                t0 = time.time()
                _, _, res = compareAuthors(["", 0, x[1], y[1], algorithm])
                t1 = time.time()
                comp_results.append(res)
                run_times.append(t1 - t0)
            pbar.update()
        pbar.close()
        tend = time.time()
        run_times = np.asarray(run_times)
        print("Average runtime to compare authors {:.3E}s".format(np.mean(run_times)))
        print("Total time to compare authors {:.3f}s".format(np.mean(tend - tstart)))
        self.assertEqual(len(comp_results), 124750)
        comp_results = []
        args = []
        tstart = time.time()

        pbar = tqdm(total=len(results[:500]), file=sys.stdout)
        for i, x in enumerate(results[:500]):
            for j, y in enumerate(results[i + 1:500]):
                args.append(["", 0, x[1], y[1], algorithm])
            pbar.update()
        pbar.close()
        with mp.Pool(4) as p:
            imap_res = list(tqdm(p.imap(compareAuthors, args), total=len(args), file=sys.stdout))
            for i in imap_res:
                comp_results.append(i)
        tend = time.time()
        print("Total time to compare authors {:.3f}s".format(np.mean(tend - tstart)))
"""
