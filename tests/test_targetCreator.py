from unittest import TestCase
import os
import json
import warnings
from src.utility_functions import loadData
from src.config_handler import ConfigHandler
from src.paper import Paper
from src.target_creator import TargetCreator
from tqdm import tqdm
from copy import deepcopy
os.chdir("..")

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestTargetCreator(TestCase):
    @ignore_warnings
    def setUp(self) -> None:

        self.config_raw = json.load(open("config.json"))
        self.config_raw["log path"] = "/tests/targetCreatorTests/logs/"
        self.config_raw["raise error"] = False
        self.config_raw["treat id different people"] = True
        self.config_raw["skip error papers"] = True
        self.log_path = self.config_raw["log path"]
        self.test_authors = [
            "hua-wu",
            "yun-chen",
            "victor-ok-li",
            "linfeng-song",
            "peng-li",
            "tatsuya-izuha",
            "yun-huang",
            "xuan-jing-huang",
            "qiang-wang"
        ]
        self.test_papers = [
            "W18-5212",
            "C18-1314",
            "P16-1159",
            "P17-1176",
            "W11-1911",
            "C14-1179",
            "P07-1089"
        ]
        self.test_multiple_auth = [
            "P17-1776",
            "C14-1179"
        ]
        self.test_non_parsed = [
            "S19-2016"
        ]
        # config = ConfigHandler(self.config_raw,"setup_test_target_creator")
        # data = loadData([ "id_to_name", "author_papers"],config.logger,config)
        self.parsed_raw = json.load(open(os.getcwd() + "/tests/authorDisambiguationTests/test_papers.json"))
        self.papers = {x: Paper(**v) for x, v in self.parsed_raw.items()}
        self.author_papers = {}
        self.id_to_name = {}
        for p, v in self.papers.items():
            for a in v.affiliations.keys():
                if a not in self.author_papers:
                    self.author_papers[a] = []
                self.author_papers[a].append(p)
            for a, n in v.authors.items():
                self.id_to_name[a] = n

    @ignore_warnings
    def test_init(self):
        print("INFO: Testing init")
        with open(os.getcwd() + self.log_path + "init.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "init")
        test_raw = TargetCreator(self.parsed_raw, self.id_to_name, self.author_papers, **config["TargetCreator"])
        for k in self.parsed_raw.keys():
            if k not in test_raw.papers:
                print("{} is missing from TargetCreator.papers when passed raw papers".format(k))
                self.fail()
        test_paper_class = TargetCreator(self.papers, self.id_to_name, self.author_papers, **config["TargetCreator"])
        for k, v in self.papers.items():
            if k not in test_paper_class.papers:
                print("{} is missing from TargetCreator.papers when passed dict of Paper classes".format(k))
                self.fail()
            if v != test_paper_class.papers[k]:
                print("Paper {} does not equal itself in TargetCreator.papers".format(k))
                self.fail()

    @ignore_warnings
    def test_updatePapers(self):
        print("INFO: Testing updatePapers")
        with open(os.getcwd() + self.log_path + "update_papers.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "update_papers")
        author_papers_copy = deepcopy(self.author_papers)
        papers_copy = {x: Paper(**v.asDict()) for x, v in self.papers.items()}
        tests = [
            ["qiang-wang","qiang-wang1",None], # No papers passed
            ['hua-wu',"hua-wu1", ['P16-1159']], # Error papers
            ['yun-chen',"yun-chen1", ['P16-1159']], # Not in paper
            ['yun-chen',"yun-chen1", ['P17-1176']],
            ['victor-ok-li',"victor-ok-li1", ['P17-1176']], # Paper already done
            ["xuan-jing-huang","fail-test",["P19-1642"]],
            ['fail-test', "yun-huang1", ['S19-2016']],
        ]

        target_creator = TargetCreator(papers_copy,self.id_to_name,author_papers_copy, **config["TargetCreator"])
        target_creator.one_per_paper = False
        target_creator.error_papers = {"P16-1159"}
        a = tests[0]
        target_creator._updatePapers(*a)
        self.assertEqual(1,len(target_creator.new_papers))
        self.assertEqual(1,len(target_creator.new_author_papers))
        self.assertTrue("qiang-wang1" in target_creator.new_author_papers)
        self.assertTrue("W19-4416" in target_creator.new_papers)
        self.assertTrue("qiang-wang1" in target_creator.new_papers["W19-4416"].authors)
        self.assertTrue("qiang-wang1" in target_creator.new_papers["W19-4416"].affiliations)


        b=tests[1]
        target_creator._updatePapers(*b)
        self.assertEqual(1, len(target_creator.new_papers))
        self.assertEqual(1, len(target_creator.new_author_papers))

        c = tests[2]
        target_creator._updatePapers(*c)
        self.assertEqual(1, len(target_creator.new_papers))
        self.assertEqual(1, len(target_creator.new_author_papers))

        d = tests[3]
        target_creator._updatePapers(*d)
        self.assertEqual(2, len(target_creator.new_papers))
        self.assertEqual(2, len(target_creator.new_author_papers))
        self.assertTrue("qiang-wang1" in target_creator.new_papers["W19-4416"].authors)
        self.assertTrue("qiang-wang1" in target_creator.new_papers["W19-4416"].affiliations)
        self.assertTrue("yun-chen1" in target_creator.new_papers["P17-1176"].authors)
        self.assertTrue("yun-chen1" in target_creator.new_papers["P17-1176"].affiliations)

        e = tests[4]
        target_creator._updatePapers(*e)
        self.assertEqual(2, len(target_creator.new_papers))
        self.assertEqual(3, len(target_creator.new_author_papers))
        self.assertTrue("yun-chen1" in target_creator.new_papers["P17-1176"].authors)
        self.assertTrue("yun-chen1" in target_creator.new_papers["P17-1176"].affiliations)
        self.assertTrue("victor-ok-li1" in target_creator.new_papers["P17-1176"].authors)
        self.assertTrue("victor-ok-li1" in target_creator.new_papers["P17-1176"].affiliations)

        f = tests[5]
        target_creator._updatePapers(*f)
        self.assertEqual(2, len(target_creator.new_papers))
        self.assertEqual(3, len(target_creator.new_author_papers))

        g = tests[6]
        target_creator._updatePapers(*g)
        self.assertEqual(2, len(target_creator.new_papers))
        self.assertEqual(3, len(target_creator.new_author_papers))

    @ignore_warnings
    def test_createTarget(self):
        with open(os.getcwd() + self.log_path + "handle_target.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "handle_target")
        target_creator = TargetCreator(self.parsed_raw, self.id_to_name, self.author_papers, **config["TargetCreator"])
        rtr = target_creator.createTarget("xuan-jing-huang")
        self.assertEqual(["1","2","3"],[x[-1] for x in rtr])
        self.assertEqual(3, len(target_creator.new_papers))
        self.assertEqual(3, len(target_creator.new_id_to_name))
        self.assertEqual(3, len(target_creator.new_author_papers))
        for k,p in target_creator.new_author_papers.items():
            for j in rtr:
                if j == k:
                    continue
                for paper in p:
                    self.assertTrue(j not in target_creator.new_papers[paper].authors)
                    self.assertTrue(j not in target_creator.new_papers[paper].affiliations)

    @ignore_warnings
    def test_fillData(self):
        with open(os.getcwd() + self.log_path + "fill_data.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "fill_data")
        target_creator = TargetCreator(self.parsed_raw, self.id_to_name, self.author_papers, **config["TargetCreator"])
        rtr = []
        for a in self.test_authors:
            rtr.extend(target_creator.createTarget(a))
        papers, auth_papers, id_to_name = target_creator.fillData()
        for a in self.test_authors:
            self.assertTrue(a not in auth_papers)
            self.assertTrue(a not in id_to_name)
        for a in rtr:
            self.assertTrue(a in auth_papers)
            self.assertTrue(a in id_to_name)

        for p in self.test_papers:
            if p not in papers:
                print(p)
            self.assertTrue(p in papers)
            found_one = False
            for a in rtr:
                if a in papers[p].authors and a in papers[p].affiliations:
                    found_one = True
                    self.assertTrue(p in auth_papers[a])
                    self.assertEqual(id_to_name[a], papers[p].authors[a])
                    break
            if not found_one:
                self.fail()