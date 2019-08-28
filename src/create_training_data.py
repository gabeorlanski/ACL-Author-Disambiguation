import fuzzysearch
import unidecode
from html import unescape
from textdistance import JaroWinkler
import numpy as np
from src.utility_functions import *
from src.paper import Paper
import multiprocessing as mp
from src.worker import Worker
import sys
from tqdm import tqdm
import time


class CreateTrainingData:
    def __init__(self, papers, incomplete_papers, special_keys=None, save_data=False, ext_directory=False, save_dir=None,
                 dif_same_ratio=2, author_cutoff=10, name_similarity_cutoff=80):
        self.papers = papers
        self.incomplete_papers = incomplete_papers
        self.special_keys = special_keys
        if not self.special_keys:
            self.special_keys = {}

        self.cores = 4
        self.save_data = save_data
        self.ext_dir = ext_directory
        self.save_dir = save_dir
        self.dif_same_ratio = dif_same_ratio
        self.author_cutoff = author_cutoff
        self.name_similarity_cutoff = name_similarity_cutoff

    def _getAlgo(self,algorithm="jaro", measure="similarity"):
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

    def __call__(self, *args, **kwargs):
        task_queue = mp.JoinableQueue()
        result_queue = mp.JoinableQueue()
        tasks, out, ignored = self._createGetAuthorInfoArgs()
        results = []
        print("INFO: Getting author info...")
        with tqdm(total=tasks, file=sys.stdout) as pbar:
            for i in out:
                results.append(self._getAuthorInfo(i))
                pbar.update()
        # processes = [Worker(task_queue, result_queue, x, self._getAuthorInfo) for x in range(self.cores)]
        # for c in processes:
        #     task_queue.put(None)
        #     c.run()
        # print("INFO: Finished getting info for {} author in {:.2f}s".format(len(results), t1 - t0))
        # results = tqdm(pool.imap_unordered(self._getAuthorInfo, auth_info_args),total=len(auth_info_args), file=sys.stdout)
        return results

    def _createGetAuthorInfoArgs(self):
        task_count = 0
        out = []
        ignored = []
        for p in self.papers.keys():

            if p in self.incomplete_papers:
                ignored.append(p)
                continue
            for author in self.papers[p].affiliations.keys():
                # task_queue.put([self.papers[p], author])
                task_count += 1
                out.append([self.papers[p], author])
        return task_count, out, ignored

    @staticmethod
    def _getAuthorInfo(args):
        paper, author = args
        pair_key = paper.pid + " " + author
        out = {
            "name": cleanName(paper.authors[author]),
            "co_authors_id": [],
            "co_authors_name": []
        }
        for a in paper.authors.keys():
            if a == author:
                continue
            out["co_authors_id"].append(a)
            out["co_authors_name"].append(cleanName(paper.authors[a]))
        aff_info = paper.affiliations[author]["affiliation"]
        email = paper.affiliations[author]["email"]
        if email:
            email = email.split("@")
            out["email_user"] = email[0]
            if len(email) == 2:
                out["email_domain"] = email[1]
            else:
                out["email_domain"] = None
        else:
            out["email_user"] = None
            out["email_domain"] = None

        try:
            out["aff_type"] = aff_info["type"][0]

        except (IndexError, KeyError) as e:
            out["aff_type"] = None

        if out["aff_type"]:
            out["aff_name"] = cleanName(aff_info["info"][out["aff_type"]][0])
            out["department"] = aff_info["info"]["department"]
        else:
            out["aff_name"] = None
            out["department"] = []
        try:
            out["address"] = aff_info["address"]
        except (IndexError, KeyError) as e:
            out["address"] = {}
        out["title"] = cleanName(paper.title)
        out["title_tokenized"] = paper.title_tokenized
        out["title_pos"] = paper.title_pos
        return pair_key, out
