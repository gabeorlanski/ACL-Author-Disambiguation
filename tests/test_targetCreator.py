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


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestTargetCreator(TestCase):
    @ignore_warnings
    def setUp(self) -> None:
        os.chdir("..")
        self.config_raw = json.load(open("config.json"))
        self.config_raw["log path"] = "/tests/targetCreatorTests/logs/"
        self.config_raw["raise error check remove"] = False
        self.config_raw["treat id different people"] = True
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
            "P18-1150",
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
            ['yun-chen',"yun-chen1", ['P17-1176']], # Will fail first, already in. Then should pass
            ['victor-ok-li',"victor-ok-li1", ['P17-1176']], # Paper already done
            ['linfeng-song',"linfeng-sing1", ['W11-1911']], # Paper is done, but not in author papers
            ['peng-li',"peng-li1", ['C14-1179']],
            ["xuan-jing-huang","fail-test",["P19-1642"]],
            ['fail-test', "yun-huang1", ['S19-2016']],
        ]

        target_creator = TargetCreator(papers_copy,self.id_to_name,author_papers_copy)
        a = tests[0]
        target_creator._updatePapers(*a)





    @ignore_warnings
    def test__handleTarget(self):
        with open(os.getcwd() + self.log_path + "handle_target.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "handle_target")
        self.fail()

    @ignore_warnings
    def test_createTarget(self):
        with open(os.getcwd() + self.log_path + "handle_target.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "handle_target")
        self.fail()
