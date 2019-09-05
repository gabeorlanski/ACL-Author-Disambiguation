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
            "heng-yu",
            "meng-zhang",
        ]
        # config = ConfigHandler(self.config_raw,"setup_test_target_creator")
        # data = loadData([ "id_to_name", "author_papers"],config.logger,config)
        self.parsed_raw = json.load(open(os.getcwd()+"/tests/authorDisambiguationTests/test_papers.json"))
        self.papers = {x:Paper(**v) for x,v in self.parsed_raw.items()}
        self.author_papers = {}
        self.id_to_name = {}
        for p, v in self.papers.items():
            for a in v.affiliations.keys():
                if a not in self.author_papers:
                    self.author_papers[a] = []
                self.author_papers[a].append(p)
            for a,n in v.authors.items():
                self.id_to_name[a] = n

    @ignore_warnings
    def test_init(self):
        print("INFO: Testing init")
        with open(os.getcwd()+self.log_path + "init.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "init")
        test_raw = TargetCreator(self.parsed_raw,self.id_to_name,self.author_papers,**config["TargetCreator"])
        for k in self.parsed_raw.keys():
            if k not in test_raw.papers:
                print("{} is missing from TargetCreator.papers when passed raw papers".format(k))
                self.fail()
        test_paper_class = TargetCreator(self.papers,self.id_to_name,self.author_papers,**config["TargetCreator"])
        for k,v in self.papers.items():
            if k not in test_paper_class.papers:
                print("{} is missing from TargetCreator.papers when passed dict of Paper classes".format(k))
                self.fail()
            if v != test_paper_class.papers[k]:
                print("Paper {} does not equal itself in TargetCreator.papers".format(k))
                self.fail()

    @ignore_warnings
    def test__updatePapersBasic(self):
        with open(os.getcwd()+self.log_path+"basic_update_paper.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw,"basic_update_paper")
        self.fail()

    @ignore_warnings
    def test__checkSafeRemove(self):
        print("INFO: Testing checkSafeRemove")
        with open(os.getcwd()+self.log_path + "check_safe_remove.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "check_safe_remove")
        author_papers_copy = deepcopy(self.author_papers)
        target_creator = TargetCreator(self.papers,self.id_to_name,self.author_papers,**config["TargetCreator"])
        for a in self.test_authors:
            can_remove, errors = target_creator._checkSafeRemove(a)
            if can_remove ==0:
                print("{} was judged as safe to remove when it should not be".format(a))
                self.fail()
            elif can_remove != -1:
                print("{} was judged as not being in any papers when it should be".format(a))
                self.fail()
            self.assertEqual([],target_creator.error_papers)
            self.assertEqual(author_papers_copy[a],target_creator.author_papers[a])

        author_papers_copy["test-a"] = deepcopy(author_papers_copy["hua-wu"])
        del author_papers_copy["hua-wu"]
        target_creator = TargetCreator(self.papers, self.id_to_name, author_papers_copy, **config["TargetCreator"])



    @ignore_warnings
    def test__handleTarget(self):
        with open(os.getcwd()+self.log_path + "handle_target.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "handle_target")
        self.fail()

    @ignore_warnings
    def test_createTarget(self):
        with open(os.getcwd()+self.log_path + "handle_target.log", "w") as f:
            pass
        config = ConfigHandler(self.config_raw, "handle_target")
        self.fail()
