import fuzzysearch
import unidecode
from html import unescape
from textdistance import JaroWinkler
import numpy as np
from src.utility_functions import *
from src.compare_authors import CompareAuthors, getAlgo
from src.paper import Paper
import time
import multiprocessing as mp
from src.worker import Worker
import sys
from tqdm import tqdm
from collections import defaultdict
import random
import gc
import pickle
import os
import logging
from hurry.filesize import size, si


# I had to put this one outside of the class because it was throwing 'cannot pickle _thread.Rlock' if it was in the
# class. I don't know why.
def checkPair(args):
    a, b, str_algorithm, special_cases, name_similarity_cutoff = args
    algorithm = getAlgo(*str_algorithm)
    if a == b:
        return None
    if a in special_cases and b in special_cases:
        return None
    a_pid, a_id = a.split(" ")
    b_pid, b_id = b.split(" ")
    if a_pid == b_pid:
        return None
    if algorithm(a_id, b_id) * 100 < name_similarity_cutoff * 100:
        return None

    if convertPaperToSortable(a_pid) < convertPaperToSortable(b_pid):
        tmp_key = a + " " + b
        pair_out = [tmp_key, a, b]
    else:
        tmp_key = b + " " + a
        pair_out = [tmp_key, b, a]
    if a_id == b_id:
        return 1, pair_out
    else:
        return 0, pair_out


def getAuthorInfo(args):
    paper, author = args
    pair_key = paper.pid + " " + author
    out = {
        "pid": paper.pid,
        "name": cleanName(paper.authors[author]),
        "co_authors_id": [],
        "co_authors_name": [],
        "co_authors_email": [],
        "co_authors_aff_type": [],
        "co_authors_aff": []
    }
    for a in paper.authors.keys():
        if a == author:
            continue
        out["co_authors_id"].append(a)
        out["co_authors_name"].append(cleanName(paper.authors[a]))
        if a in paper.affiliations:
            auth_aff = paper.affiliations[a]
            if auth_aff["email"]:
                out["co_authors_email"].append(auth_aff["email"].split("@"))
            else:
                out["co_authors_email"].append([None, None])
            try:
                auth_aff_type = auth_aff["affiliation"]["type"][0]
            except:
                auth_aff_type = None
            if auth_aff_type:
                out["co_authors_aff"].append(cleanName(auth_aff["affiliation"]["info"][auth_aff_type][0]))
            else:
                out["co_authors_aff"].append(None)
            out["co_authors_aff_type"].append(auth_aff_type)
        else:
            out["co_authors_aff"].append(None)
            out["co_authors_email"].append([None, None])
            out["co_authors_aff_type"].append(None)
            out["co_authors_aff"].append(None)
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
        if out["address"]["settlement"]:
            out["address"]["settlement"] = cleanName(out["address"]["settlement"])
        if out["address"]["country"]:
            out["address"]["country"] = cleanName(out["address"]["country"])
    except (IndexError, KeyError) as e:
        out["address"] = {}
    out["title"] = cleanName(paper.title)
    out["title_tokenized"] = paper.title_tokenized
    out["citations"] = paper.citations
    out["citations_tokenized"] = paper.citations_tokenized
    out["sections"] = paper.sections
    out["sections_tokenized"] = paper.sections_tokenized
    return pair_key, out


class CreateTrainingData:
    def __init__(self, papers, incomplete_papers, special_keys=None, save_data=False, ext_directory=False,
                 save_dir=None, diff_same_ratio=2.0, author_cutoff=10, name_similarity_cutoff=.6,
                 pair_distribution="random", separate_chars=1, separate_words=1, algorithm="jaro-similarity",
                 exclude=None, rand_seed=None, cores=4, batch_size=25000, allow_exact_special=False,
                 min_batch_len=100000, file_log_level=logging.DEBUG, console_log_level=logging.WARNING, log_format=None,
                 log_path=None, DEBUG_MODE=False, drop_null_authors=True, print_compare_stats=False, compare_args=None,
                 compare_batch_size=2000, remove_single_author=False, require_exact_match=False):
        """
        Initialize the class
        :param papers: The parsed papers you want to use (dict of Paper objects)
        :param incomplete_papers: Papers you want to exclude (list of paper ids)
        :param special_keys: Any special keys you want to guarantee are in the training data (list of ids,
        defaults to empty list)
        :param save_data: Save data for later use (Bool, default is false)
        :param ext_directory: Save data into directories based on their file type (Bool, defaults to false)
        :param save_dir: Directory to save data (str, defaults to none)
        :param diff_same_ratio: Ratio of same pairs to different pairs, and vice-versa. (float, default is 2)
        :param author_cutoff: Cutoff authors based on paper count (int, default is 10)
        :param name_similarity_cutoff: Exclude pairs if their names arent similar enough (float, default is .6)
        :param pair_distribution: how to distribute pairs to meet ratio Options are 'random' and 'sim distribution'.
        sim distribution tries to get an even number of pairs based on how similar their names are (default is random)
        :param separate_chars: How many chars you want to use in the pair dict (int, default is 1)
        :param separate_words: # of words to use in the pair dict (int, default is 1)
        :param algorithm: string similarity algorithm(str in format of 'algorithm name-measure' default is
        'jaro-similarity')
        :param exclude: authors to exclude(list of ids, default is empty list)
        :param rand_seed: Random seed
        :param cores: Cores to use (int, default is 4)
        :param allow_exact_special: allow ids that are exactly equal to special cases (bool, default is false)
        :param batch_size: Size of batches (int, default is 25000)
        :param min_batch_len: Minimum number of combinations to use batches (int, default is 100,000)
        :param file_log_level: logging level to file (logging.levels, default is debug)
        :param console_log_level: logging level to console (logging.levels, default is debug)
        :param log_format: format of log messages (str)
        :param log_path: path to log files(str, default is '/logs/preprocess_data.log'
        :param DEBUG_MODE: debugging mode (bool, default is false)
        :param drop_null_authors: drop authors with either no email or no affiliation (bool, default is True)
        :param print_compare_stats: print the indepth stats of comparisons. This WILL slow
        down the program by a lot(bool, default is False)
        :param compare_args: dict of arguments to pass to compareAuthors
        :param compare_batch_size: size of batches for comparing authors, only has an effect when cores > 1
        :param remove_single_author: Remove papers with only 1 author
        :param require_exact_match: If special cases must be exact match
        """
        if compare_args is None:
            compare_args = {}
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/preprocess_data.log"
        self.papers = papers
        self.incomplete_papers = incomplete_papers
        self.special_keys = special_keys
        if not self.special_keys:
            self.special_keys = []
        self.cores = cores
        self.save_data = save_data
        self.ext_directory = ext_directory
        self.save_dir = save_dir
        self.dif_same_ratio = diff_same_ratio
        self.author_cutoff = author_cutoff
        self.name_similarity_cutoff = name_similarity_cutoff
        self.author_papers = defaultdict(list)
        self.pair_distribution = pair_distribution
        if rand_seed:
            random.seed(rand_seed)
        self.separate_chars = separate_chars
        self.separate_words = separate_words
        self.exclude = exclude if exclude else []
        self.batch_size = batch_size
        self.allow_exact_special = allow_exact_special
        self.algorithm = algorithm.split("-")
        self.min_batch_len = min_batch_len
        self.json_path = self.save_dir
        self.csv_path = self.save_dir
        self.txt_path = self.save_dir
        self.pickle_path = self.save_dir
        self.debug_mode = DEBUG_MODE
        self.console_log_level = console_log_level
        self.drop_null_authors = drop_null_authors
        self.compare_args = compare_args
        self.compare_args["str_algorithm"] = algorithm.split("-")
        self.compare_batch_size = compare_batch_size
        if self.ext_directory:
            self.json_path = self.json_path + "/json"
            self.csv_path = self.csv_path + "/csv"
            self.txt_path = self.txt_path + "/txt"
            self.pickle_path = self.pickle_path + "/pickle"
            if not os.path.exists(self.json_path):
                os.mkdir(self.json_path)
            if not os.path.exists(self.csv_path):
                os.mkdir(self.csv_path)
            if not os.path.exists(self.txt_path):
                os.mkdir(self.txt_path)
            if not os.path.exists(self.pickle_path):
                os.mkdir(self.pickle_path)

        self.logger = createLogger("create_training_data", log_path, log_format, console_log_level,
                                   file_log_level)

        self.print_compare_stats = print_compare_stats
        self.remove_single_author = remove_single_author
        self.require_exact_match = require_exact_match

    def __call__(self, pairs_to_use=None, authors_to_use=None, debug_retrieve_info=None, get_info_all=False,
                 debug_asserts=False):
        total_run_start = time.time()
        if pairs_to_use and authors_to_use:
            self.logger.warning(
                "Both pairs_to_use and authors_to_use were passed, pairs_to_use will override authors_to_use")
        if authors_to_use is None:
            authors_to_use = []
        override_pairs_to_use = False
        if pairs_to_use is None:
            pairs_to_use = []
            override_pairs_to_use = True
        tasks, out, ignored, excluded = self._populateConstants()
        results = []
        paper_auth_info = {}

        """
        Initialize data
        """
        printLogToConsole(self.console_log_level, "Getting author info", logging.INFO)
        self.logger.log(logging.INFO, "Getting author info")
        below_cutoff = 0
        with tqdm(total=tasks, file=sys.stdout) as pbar:
            for i in out:
                add_author = False
                if authors_to_use and i[1] in authors_to_use:
                    add_author = True
                elif len(self.author_papers[i[1]]) >= self.author_cutoff or i[1] in self.special_keys:
                    add_author = True
                elif len(self.author_papers[i[1]]) < self.author_cutoff:
                    below_cutoff += 1
                if i[1] in self.special_keys:
                    add_author = True

                if add_author:
                    pair_key, res = getAuthorInfo(i)
                    # results.append((pair_key, res))
                    paper_auth_info[pair_key] = res
                pbar.update()
            pbar.close()
        self.logger.debug("{} Authors below cutoff".format(below_cutoff))

        """
        Separate the authors by the first char(s) that appear in their ids, to reduce the number of pointless pairs
        """
        printLogToConsole(self.console_log_level, "Separating keys by chars in name", logging.INFO)
        self.logger.log(logging.INFO, "Separating keys by chars in name")
        separated_keys = self._createPairDict(list(paper_auth_info.keys()), self.separate_chars, self.separate_words)

        """
        Create the pairs needed and put the special cases into their own arrays, then select the pairs to use based 
        on the ratio defined in initialization to ensure a controlled number of same to different. This DOES NOT 
        occur to any special cases. If pairs_to_use is defined, skip and use predefined pairs
        """
        if not pairs_to_use:
            printLogToConsole(self.console_log_level, "Creating pairs", logging.INFO)
            self.logger.log(logging.INFO, "Creating pairs")
            same, diff, special_same, special_diff = self._prepareData(separated_keys, paper_auth_info, self.algorithm)
            self.logger.debug("len(same) = {}".format(len(same)))
            self.logger.debug("len(different) = {}".format(len(diff)))
            self.logger.debug("len(special_same) = {}".format(len(special_same)))
            self.logger.debug("len(special_different) = {}".format(len(special_diff)))
            if not get_info_all:
                self.logger.debug("Splitting pairs")
                same, diff = self._selectPairsToUse(same, diff)
            else:
                self.logger.debug("Getting all info")

            pairs_to_use = [*same, *diff, *special_same, *special_diff]
        else:
            printLogToConsole(self.console_log_level, "Using passed pairs", logging.INFO)
            self.logger.log(logging.INFO, "Using passed pairs")
            same = []
            diff = []
            special_same = []
            special_diff = []
            for t, pair_data in pairs_to_use:
                is_special = False
                for special_case in self.special_keys:
                    if special_case in pair_data[1] or special_case in pair_data[2]:
                        is_special = True
                if t == 1:
                    if is_special:
                        special_same.append([t, pair_data])
                    else:
                        same.append([t, pair_data])
                elif t == 0:
                    if is_special:
                        special_diff.append([t, pair_data])
                    else:
                        diff.append([t, pair_data])

        """
        Take the pairs and get the info needed for them. This is done here in order to save runtime memory
        """
        to_use = []
        printLogToConsole(self.console_log_level, "Retrieving info for pairs", logging.INFO)
        self.logger.log(logging.INFO, "Retrieving info for pairs")
        for p in pairs_to_use:
            try:
                tag, pair_info = p
            except ValueError as e:
                self.logger.error("Error raised when retrieving info for pairs")
                self.logger.error(
                    "Issue with value unpacking for p when iterating over pairs_to_use. expected 2 got {}".format(
                        len(p)))
                self.logger.error("p: {}".format(p))
                self.logger.exception(e)
                raise e  # just to make warnings stop

            try:
                key, a, b = pair_info
            except ValueError as e:
                self.logger.error("Error raised when unpacking pair info")
                self.logger.error("Issue with value unpacking for pair_info. expected 3 got {}".format(len(pair_info)))
                self.logger.error("pair_info: {}".format(pair_info))
                self.logger.exception(e)
                raise e
            to_use.append([key, tag, paper_auth_info[a], paper_auth_info[b]])
        if debug_retrieve_info:
            return to_use
        random.shuffle(to_use)

        """
        Compare those pairs, print out result stats, and pickle the data
        """
        comparator = CompareAuthors(**self.compare_args)
        printLogToConsole(self.console_log_level, "Comparing authors", logging.INFO)
        self.logger.log(logging.INFO, "Comparing authors")
        if self.cores == 1 or len(to_use) < 20000:
            pbar = tqdm(total=len(to_use), file=sys.stdout)
            for i in to_use:
                results.append(comparator(i))
                pbar.update()
            pbar.close()
        else:
            printLogToConsole(self.console_log_level, "Comparing {} pairs in parallel".format(len(to_use)),
                              logging.INFO)
            self.logger.info("Comparing {} pairs in parallel".format(len(to_use)))

            batches = chunks(to_use, self.compare_batch_size)
            batch_count = len(to_use) // self.compare_batch_size
            if len(to_use) % self.compare_batch_size != 0:
                batch_count += 1
            self.logger.debug("{} batches".format(batch_count))

            with mp.Pool(self.cores) as Pool:
                imap_results = list(
                    tqdm(Pool.imap_unordered(comparator.processBatch, batches), total=batch_count, file=sys.stdout))

            self.logger.debug("Combining results from pool")
            for res in imap_results:
                results.extend(res)
        total_run_end = time.time()
        hours, rem = divmod(total_run_end - total_run_start, 3600)
        minutes, seconds = divmod(rem, 60)
        stats = [
            ["Total Pairs Used", len(to_use)],
            ["Same", len(same)],
            ["Different", len(diff)],
            ["Special Same", len(special_same)],
            ["Special Different", len(special_diff)],

        ]
        printStats("Results", stats, line_adaptive=True)
        printLogToConsole(self.console_log_level,
                          "Total Run time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
                          logging.INFO)
        self.logger.info("Total Run Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        if self.save_data:
            printLogToConsole(self.console_log_level, "Pickling results", logging.INFO)
            with open("tagged_pairs.pickle", "wb") as f:
                pickle.dump(results, f)

    def _populateConstants(self):
        task_count = 0
        out = []
        ignored = []
        excluded = []
        dropped_null = 0
        special_case_count = 0
        dropped = []
        pbar = tqdm(total=len(self.papers), file=sys.stdout)
        self.logger.debug("{} total papers".format(len(self.papers)))
        for p, info in self.papers.items():
            if p in self.incomplete_papers:
                ignored.append(p)
                pbar.update()
                continue
            for author in info.affiliations.keys():
                if author in self.exclude or (not self.allow_exact_special and any([x for x in self.special_keys if
                                                                                    x == author])):
                    pbar.update()
                    excluded.append((p, author))
                    continue
                is_special = False
                if any([x for x in self.special_keys if x in author]):
                    is_special = True
                    special_case_count += 1
                if self.drop_null_authors:
                    has_null = False
                    try:
                        test_type = info.affiliations[author]["affiliation"]["type"][0]
                    except IndexError as e:
                        has_null = True

                    if not info.affiliations[author]["email"]:
                        has_null = True
                    if has_null and not is_special:
                        dropped_null += 1
                        dropped.append([p, author])
                        pbar.update()
                        continue

                task_count += 1
                out.append([info, author])
                self.author_papers[author].append(p)
            pbar.update()
        pbar.close()
        self.logger.debug("{} Authors excluded".format(len(excluded)))
        self.logger.debug("{} Papers ignored".format(len(ignored)))
        self.logger.debug("{} special cases".format(special_case_count))
        self.logger.debug("Dropped {} author cases for having null in either affiliation or email".format(
            dropped_null))
        return task_count, out, ignored, excluded

    @staticmethod
    def _createPairDict(pair_keys, char_count=1, word_count=1):
        out = defaultdict(list)
        for k in pair_keys:
            p, a = k.split()
            a_split = a.split("-")
            first_word = a_split[:word_count]
            out[" ".join(first_word)[:char_count]].append(k)
        return out

    def _getSpecialCases(self, separated):
        special_cases = {}
        for k in self.special_keys:
            special_split = " ".join(k.split("-")[:self.separate_words])
            special_split = special_split[:self.separate_chars]
            self.logger.debug("{} special key = {}".format(k, special_split))
            if special_split not in separated:
                continue
            if special_split not in special_cases:
                special_cases[special_split] = []
            for a in separated[special_split]:
                if k in a:
                    if (not self.allow_exact_special and k != a) or self.allow_exact_special:
                        if self.require_exact_match:
                            if k == a:
                                special_cases[special_split].append(a)
                        else:
                            special_cases[special_split].append(a)
        for k in special_cases.keys():
            self.logger.debug("special_cases[{}] len = {}".format(k, len(special_cases[k])))
        return special_cases

    def _makeCombinations(self, i, special_cases=None, use_cutoff=True):
        if not special_cases:
            special_cases = []
        self.logger.debug("len(special_cases)={}".format(len(special_cases)))
        infos = {x[0]: x[1] for x in i}
        keys = list(infos.keys())
        same = {}
        different = {}
        name_cutoff = self.name_similarity_cutoff if use_cutoff else 0
        combinations = []
        pair_creator_pbar = tqdm(total=int(ncr(len(keys), 2)), file=sys.stdout)
        max_size_allocate = 0
        estimated_size_to_allocate = 0
        special_cases_combos = []
        for i, a in enumerate(keys):
            for j, b in enumerate(keys[1 + i:]):
                a_paper, a_id = a.split(" ")
                b_paper, b_id = b.split(" ")
                if a_id in special_cases and b_id in special_cases:
                    special_cases_combos.append([a, b, self.algorithm, special_cases, name_cutoff])
                    continue
                combo_to_add = [a, b, self.algorithm, special_cases, name_cutoff]
                combo_size = sys.getsizeof(combo_to_add)
                combinations.append(combo_to_add)
                max_size_allocate = max(max_size_allocate, combo_size)
                estimated_size_to_allocate += combo_size
                pair_creator_pbar.update()
        pair_creator_pbar.close()
        self.logger.log(logging.DEBUG, "{} combinations".format(len(combinations)))
        self.logger.debug("{} special combinations".format(len(special_cases_combos)))
        self.logger.log(logging.DEBUG, "Max combination size {} bytes".format(max_size_allocate))
        self.logger.log(logging.DEBUG, "Size of combinations {}".format(size(estimated_size_to_allocate, si)))

        printLogToConsole(self.console_log_level, "Removing pairs that are not valid", logging.INFO)
        self.logger.log(logging.INFO, "Removing pairs that are not valid")
        total_combinations = len(combinations)
        if self.cores == 1 or total_combinations < self.min_batch_len:
            if total_combinations < self.min_batch_len:
                self.logger.debug("total combinations is less than min batch length({} < {})".format(
                    total_combinations, self.min_batch_len))
            pbar = tqdm(total=total_combinations, file=sys.stdout)
            for combo in combinations:
                res = checkPair(combo)
                if res:
                    tag, res = res
                    if tag == 1:
                        same[res[0]] = res
                    else:
                        different[res[0]] = res
                pbar.update()
            pbar.close()
        else:
            # Put the pairs into batches so they can be used in parallel, otherwise the overhead is too much
            batches = chunks(combinations, self.batch_size)
            batch_count = total_combinations // self.batch_size
            tmp_same = []
            tmp_different = []
            possible_errors = 0
            if total_combinations % self.batch_size != 0:
                batch_count += 1
            self.logger.debug("{} total batches".format(batch_count))
            imap_results = []
            t0 = time.time()
            with mp.Pool(self.cores) as Pool:
                try:
                    imap_results = list(
                        tqdm(Pool.imap_unordered(self._batchCheckPair, batches), total=batch_count, file=sys.stdout))
                except Exception as e:
                    print()
                    self.logger.exception("Exception raised when putting batches into pool", exc_info=e)
                    raise e
            t1 = time.time()
            self.logger.debug("{:.2f} combos/second".format(total_combinations / (t1 - t0)))
            if not imap_results:
                printLogToConsole(self.console_log_level, "imap_results is empty", logging.ERROR)
                self.logger.log(logging.ERROR, "imap_results is empty")
                return [], []
            for s, d in imap_results:
                tmp_same.extend(s)
                tmp_different.extend(d)

                # tmp_same = list(set(tmp_same))
                # tmp_different = list(set(tmp_different))
            printLogToConsole(self.console_log_level, "Combining results from pool", logging.INFO)
            self.logger.log(logging.INFO, "Combining results from pool")
            for i in tmp_same:
                if i[0] in same:
                    possible_errors += 1
                same[i[0]] = i
            for i in tmp_different:
                if i[0] in different:
                    possible_errors += 1
                different[i[0]] = i
            printLogToConsole(self.console_log_level, "{} overlapping keys".format(possible_errors), logging.DEBUG)
            self.logger.log(logging.DEBUG, "{} overlapping keys".format(possible_errors))
        gc.collect()
        self.logger.log(logging.DEBUG, "Removed {} pairs".format(total_combinations - (len(same) + len(different))))
        return [v for _, v in same.items()], [v for _, v in different.items()]

    @staticmethod
    def _batchCheckPair(args):
        same = []
        different = []
        for i in args:
            res = checkPair(i)
            if res:
                tag, res = res
                if tag == 1:
                    same.append(res)
                else:
                    different.append(res)

        # Most likely will run out of memory with this
        gc.collect()

        return same, different

    @staticmethod
    def _convertToArg(pair_args, tag, algorithm):
        pair_key, a, b = pair_args
        return pair_key, tag, a, b, algorithm

    def _prepareData(self, separated, paper_auth_info, algorithm):
        special_cases_dict = self._getSpecialCases(separated)
        same = []
        different = []
        special_same = []
        special_diff = []
        sorted_keys = sorted(list(separated.keys()))

        for k in sorted_keys:
            info = separated[k]
            printLogToConsole(self.console_log_level, "Creating pairs for authors starting with {}".format(k),
                              logging.INFO)
            self.logger.log(logging.INFO, "Creating pairs for authors starting with {}".format(k))
            special_cases = None
            if k in special_cases_dict:
                special_cases = special_cases_dict[k]

            auth_info = [(x, paper_auth_info[x]) for x in info]
            tmp_same, tmp_diff = self._makeCombinations(auth_info, special_cases)
            gc.collect()
            self.logger.debug("{} pairs to add to same".format(len(tmp_same)))
            self.logger.debug("{} pairs to add to different".format(len(tmp_diff)))
            same.extend([[1, p] for p in tmp_same])
            different.extend([[0, p] for p in tmp_diff])
            self.logger.debug("{} same pairs".format(len(same)))
            self.logger.debug("{} different pairs".format(len(different)))
        printLogToConsole(self.console_log_level, "Handling special cases", logging.INFO)
        self.logger.log(logging.INFO, "Handling special cases")
        for k, info in special_cases_dict.items():
            printLogToConsole(self.console_log_level, "Creating pairs for special cases that start with {}".format(k),
                              logging.INFO)
            self.logger.log(logging.INFO, "Creating pairs for special cases that start with {}".format(k))
            auth_info = [(x, paper_auth_info[x]) for x in info]
            tmp_same, tmp_diff = self._makeCombinations(auth_info, algorithm, use_cutoff=False)
            special_same.extend([[1, p] for p in tmp_same])
            special_diff.extend([[0, p] for p in tmp_diff])

        return same, different, special_same, special_diff

    def _selectPairsToUse(self, same, diff):
        printLogToConsole(self.console_log_level, "Selecting pairs to use", logging.DEBUG)
        self.logger.log(logging.INFO, "Selecting pairs to use")
        self.logger.debug("len(same) -> {}".format(len(same)))
        self.logger.debug("len(different) -> {}".format(len(diff)))
        len_same = len(same)
        len_diff = len(diff)
        if len_same > len_diff:
            pair_count = int(len_diff * self.dif_same_ratio)
        else:
            pair_count = int(len_same * self.dif_same_ratio)
        self.logger.debug("pair_count -> {}".format(pair_count))

        if self.pair_distribution == "similarity":
            printLogToConsole(self.console_log_level, "Using similarity distribution", logging.INFO)
            self.logger.log(logging.INFO, "Using similarity distribution")
            printLogToConsole(self.console_log_level, "Similarity distribution is not implemented yet",
                              logging.CRITICAL)
            self.logger.log(logging.ERROR, "Similarity distribution is not implemented yet")
            # TODO: Implement similarity distribution
            raise ValueError("Similarity distribution is not implemented yet")
        elif self.pair_distribution == "random":
            printLogToConsole(self.console_log_level, "Using random selection", logging.INFO)
            self.logger.log(logging.INFO, "Using random selection")
            try:
                out_same = random.sample(same, pair_count)
            except:
                out_same = same[:pair_count]
            try:
                out_diff = random.sample(diff, pair_count)
            except:
                out_diff = diff[:pair_count]
            return out_same, out_diff

    def parameterDict(self) -> dict:
        out = {
            "separate_chars": self.separate_chars,
            "separate words": self.separate_words,
            "name cutoff": self.name_similarity_cutoff,
            "paper_cutoff": self.author_cutoff,
            "pair distribution": self.pair_distribution,
            "allow_exact_special": self.allow_exact_special,
            "algorithm": "-".join(self.algorithm),
            "drop_null_authors": self.drop_null_authors,
            "remove_single_author": self.remove_single_author
        }

        return out
