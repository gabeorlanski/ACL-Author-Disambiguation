from tqdm import tqdm, trange
import os
import json
import logging
from src.utility_functions import cleanName, createLogger, printLogToConsole, nameFromDict
from src.create_training_data import getAuthorInfo
from src.compare_authors import CompareAuthors, getAlgo
from src.paper import Paper
import numpy as np
from collections import defaultdict, Counter
import sys
import pickle
from copy import deepcopy
import multiprocessing as mp
import gc


class AuthorDisambiguation:
    parameters = dict(
        threshold=[.1, "Minimum similarity threshold for considering an author_id as the same as the target"],
        name_similarity_cutoff=[.9, "Minimum string similarity of the other name when "],
        str_algorithm=["jaro-similarity", ""],
        model_name=["VC1", "Name of the model to use"],
        model_path=["", "Path to the model, defaults to 'cwd+/models/'"],
        create_new_author=[False, "Create new authors if no similar authors are found"],
        compare_cutoff=[3, "Minimum papers for an id to compare it to the target"],
        tie_breaker=["max", "Method to use for breaking ties when more than one id is above the threshold"],
        sim_overrides=[False, "Name similarity score overrides initials not being the same. For example: \nAuthor a id = "
                              "'auth-a-org' and name is 'John (Williams) Doe\nTarget is 'auth-a' and name is 'John Doe'\n With "
                              "true it will override the difference in initials by using the similarity of the names"],
        allow_authors_not_in_override=[True, "When passing targets to call, Disable allowing authors who do not have "
                                             "predefined authors to compare"],
        same_paper_diff_people=[True, "Disable removing ids who share papers with the target"],
        use_probabilities=[False, "Use probabilities instead of predictions, only works if the model allows this"]
    )

    def __init__(self, papers=None, author_papers=None, compare_args=None, id_to_name=None,
                 console_log_level=logging.ERROR, file_log_level=logging.DEBUG, log_format=None, log_path=None,
                 save_data=False, ext_directory=False, save_path=None, threshold=.75, name_similarity_cutoff=.9,
                 str_algorithm="jaro-similarity", model=None, model_name="VC1", model_path=None,
                 create_new_author=False, compare_cutoff=3, tie_breaker="max", cores=4, DEBUG_MODE=False,
                 sim_overrides=False, allow_authors_not_in_override=True, same_paper_diff_people=True, use_probabilities=False):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/disambiguation.log"
        self.logger = createLogger("author_disambiguation", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.model = model
        self.model_name = model_name
        if self.model is None:
            if not model_path:
                model_path = os.getcwd()
            self.model = pickle.load(open("{}/models/{}/model.pickle".format(model_path, model_name), "rb"))
        try:
            if self.model.voting == "hard" and use_probabilities:
                self.logger.warning("hard voting does not support probabilities")
                self.use_probabilities = False
            else:
                self.use_probabilities = use_probabilities
        except Exception as e:
            self.logger.debug("model does not have voting")
            self.use_probabilities = False

        if not DEBUG_MODE:
            # Argument validation
            if compare_args and not isinstance(compare_args, dict):
                self.logger.error("passed compare_args is not valid")
                self.logger.exception(TypeError("compare_args is not a dict"))
                raise TypeError("compare_args is not a dict")
            elif not compare_args:
                self.logger.error("passed compare_args is not valid")
                self.logger.exception(ValueError("compare_args is None"))
                raise ValueError("compare_args is None")
            else:
                self.compare_args = compare_args

            if author_papers and (not isinstance(author_papers, dict) and not isinstance(author_papers, defaultdict)):
                self.logger.error("passed author_papers is not valid")
                self.logger.error("type is {}".format(type(author_papers)))
                self.logger.exception(TypeError("author_papers is not a dict"))
                raise TypeError("author_papers is not a dict")
            elif not author_papers:
                author_papers, status, error_msg = self._findData("author_papers.json")
                if status != 0:
                    self.logger.error(
                        "passed author_papers is not valid and could not find the file author_papers.json")
                    self.logger.error("self._findData(\"author_papers.json\") returned error {}".format(error_msg))
                    self.logger.exception(ValueError("No valid author_papers found"))
                    raise ValueError("No valid author_papers found")
                else:
                    self.author_papers = deepcopy(author_papers)
            else:
                self.author_papers = deepcopy(author_papers)

            if papers and not isinstance(papers, dict):
                self.logger.error("passed papers is not valid")
                self.logger.exception(TypeError("papers is not a dict"))
                raise TypeError("papers is not a dict")
            elif not papers:
                papers, status, error_msg = self._findData("parsed_papers.json")
                if status != 0:
                    self.logger.error("passed papers is not valid and could not find the file parsed_papers.json")
                    self.logger.error("self._findData(\"parsed_papers.json\") returned error {}".format(error_msg))
                    self.logger.exception(ValueError("No valid parsed_papers found"))
                    raise ValueError("No valid parsed_papers found")
                else:
                    if len(papers) == 0:
                        self.logger.exception(ValueError("Found papers is empty"))
                        raise ValueError("Found papers is empty")
                    self.logger.debug("Converting papers from dict to Paper object")
                    self.papers = {}
                    for k, info in papers.items():
                        self.papers[k] = Paper(**info)

            else:
                if len(papers) == 0:
                    self.logger.exception(ValueError("Passed papers is empty"))
                    raise ValueError("Passed papers is empty")
                test_key = list(papers.keys())[0]
                if isinstance(test_key, dict):
                    self.papers = {}
                    for k, info in papers.items():
                        try:
                            self.papers[k] = Paper(**info)
                        except Exception as e:
                            self.logger.error("Exception raised when converting paper dicts to Paper")
                            self.logger.error("k={}".format(k))
                            self.logger.error("info={}".format(info))
                            self.logger.exception(e)
                            raise e
                else:
                    self.papers = papers

            if id_to_name and not isinstance(id_to_name, dict):
                self.logger.error("passed id_to_name is not valid")
                self.logger.exception(TypeError("id_to_name is not a dict"))
                raise TypeError("id_to_name is not a dict")
            elif not id_to_name:
                id_to_name, status, error_msg = self._findData("id_to_name.json")
                if status != 0:
                    self.logger.error("passed id_to_name is not valid and could not find the file parsed_papers.json")
                    self.logger.error("self._findData(\"id_to_name.json\") returned error {}".format(error_msg))
                    self.logger.exception(ValueError("No valid id_to_name found"))
                    raise ValueError("No valid id_to_name found")
                else:
                    if len(id_to_name) == 0:
                        self.logger.exception(ValueError("Found id_to_name is empty"))
                        raise ValueError("Found id_to_name is empty")
                    self.id_to_name = id_to_name

            else:
                if len(id_to_name) == 0:
                    self.logger.exception(ValueError("Passed id_to_name is empty"))
                    raise ValueError("Passed id_to_name is empty")
                self.id_to_name = id_to_name
        else:
            printLogToConsole(self.console_log_level, "RUNNING IN DEBUG_MODE!", logging.WARNING)
            self.logger.warning("Running in DEBUG_MODE")
            self.id_to_name = id_to_name if id_to_name else {}
            self.papers = papers if papers else {}
            self.compare_args = compare_args if compare_args else {}
            self.author_papers = author_papers if author_papers else {}
        self.compare_terms = len(CompareAuthors.compare_terms)
        self.save_data = save_data
        self.save_dir = save_path
        self.ext_directory = ext_directory
        self.threshold = threshold
        self.name_similarity_cutoff = name_similarity_cutoff
        algo_name, measure = str_algorithm.split("-")
        self.author_name = {x: nameFromDict(self.id_to_name[x]) for x in self.id_to_name.keys()}
        self.cores = cores
        self.str_algorithm = getAlgo(algo_name, measure)
        self.create_new_author = create_new_author
        self.compare_cutoff = compare_cutoff
        self.tie_breaker = tie_breaker
        self.sim_overrides = sim_overrides
        self.allow_authors_not_in_override = allow_authors_not_in_override
        self.same_paper_diff_people = same_paper_diff_people
        self.logger.debug("AuthorDisambiguation initialized with arguments:")
        self.logger.debug("\tcompare_args={}".format(list(self.compare_args.keys())))
        self.logger.debug("\talgorithm={}".format(algo_name))
        self.logger.debug("\tmeasure={}".format(measure))
        self.logger.debug("\tthreshold={}".format(threshold))
        self.logger.debug("\tname_similarity_cutoff={}".format(name_similarity_cutoff))
        self.logger.debug("\tunique authors={}".format(len(self.author_papers)))
        self.logger.debug("\tcompare_cutoff={}".format(self.compare_cutoff))
        self.logger.debug("\ttie_breaker={}".format(self.tie_breaker))
        self.logger.debug("\tsim_overrides={}".format(self.sim_overrides))
        self.logger.debug("\tsame_paper_diff_people={}".format(self.same_paper_diff_people))
        self.logger.debug("\tuse_probabilities={}".format(self.use_probabilities))
        if self.compare_cutoff != 3:
            self.logger.warning("Non-default value for compare_cutoff, currently this is not implemented")

    def _findData(self, file_name):
        file_ext = file_name.split(".")[-1]
        self.logger.debug("Looking for file {}".format(file_name))
        cwd = os.getcwd()
        if not os.path.isdir(cwd + "/data"):
            return None, -1, "{} is not a directory".format(cwd + "/data")
        cwd = cwd + "/data"
        file_path = cwd + "/" + file_name

        self.logger.debug("Checking if {} exists".format(file_path))
        if os.path.exists(file_name):
            return self._parseFoundFile(file_path, file_ext)

        self.logger.debug("{} does not exist, moving on to checking if {} is a directory".format(file_path, file_ext))
        if not os.path.isdir(cwd + "/" + file_ext):
            return None, -1, "{} is not a directory".format(cwd + "/" + file_ext)
        cwd = cwd + "/" + file_ext

        self.logger.debug("{} is a valid directory, not checking if file exists".format(cwd))
        if os.path.exists(cwd + "/" + file_name):
            self.logger.debug("Found {} in subdirectory {}".format(file_name, file_ext))
            return self._parseFoundFile(cwd + "/" + file_name, file_ext)

        self.logger.debug("Cound not find file")
        return None, -1, "{} not found in any subdirectory".format(file_name)

    @staticmethod
    def _parseFoundFile(file_path, ext):
        if ext == "json":
            out = json.load(open(file_path))
        elif ext == "xml":
            out = open(file_path, "rb")
        elif ext == "csv":
            out = [x.strip().split(",") for x in open(file_path).readlines()]
        elif ext == "txt":
            out = [x.strip() for x in open(file_path).readlines()]
        else:
            return None, -1, "{} does not have a supported extensions"
        return out, 0, ""

    def _getAuthorInfos(self, authors) -> (dict, int, int):
        out = {}
        printLogToConsole(self.console_log_level, "Getting author info for specified authors", logging.INFO)
        self.logger.info("Getting author info for specified authors")
        self.logger.debug("authors={}".format(authors))
        error_authors = 0
        error_papers = 0
        pbar = tqdm(total=len(authors), file=sys.stdout)
        for a in authors:
            if a not in self.author_papers:
                pbar.update()
                self.logger.warning("{} is not in self.author_papers".format(a))
                error_authors += 1
                continue
            for p in self.author_papers[a]:
                if p not in self.papers:
                    self.logger.debug("{} not in self.papers".format(p))
                    error_papers += 1

                    continue
                auth_key, auth_info = getAuthorInfo([self.papers[p], a])
                out[auth_key] = auth_info
            pbar.update()
        pbar.close()
        self.logger.debug("len(out)={}".format(len(out)))
        self.logger.debug("error_authors={}".format(error_authors))
        self.logger.debug("error_papers={}".format(error_papers))
        return out, error_authors, error_papers

    @staticmethod
    def _getSimilarAuthors(args):
        target_id, target_author, author_name, str_algorithm, name_similarity_cutoff, sim_overrides = args
        out = []

        target_initials = [w[0] for w in target_author.split()]

        warnings = []
        debug = []
        authors_use = []
        for _id, name in author_name.items():
            first_letter = name[0].lower()
            if first_letter == target_author[0].lower():
                authors_use.append([_id, name])
        debug.append("{} authors with the same first letter as {}".format(len(authors_use), target_id))
        for _id, name in authors_use:
            cleaned_name = cleanName(name).lower()

            tmp_id = "-".join(_id.split("-")[:len(target_initials)])
            pass_sim_test = False

            if str_algorithm(target_id, tmp_id) * 100 >= name_similarity_cutoff * 100:
                pass_sim_test = True
            override_with_sim = sim_overrides and pass_sim_test

            # Do not override the first name check b/c the first name check prevents authors with the targets name in
            # their name from being used.
            # For example:
            #   target is yang-liu
            #   the author it is looking at is luyang-liu.
            #   It would pass the similarity test, but we know it is not the same because the first name is
            if str_algorithm(cleaned_name.split()[0],
                             target_author.split()[0]) * 100 < name_similarity_cutoff * 100:
                if pass_sim_test:
                    warnings.append(
                        "{} passed the similarity test, but does not have the same first name".format(_id))
                    warnings.append("author name ={}".format(name))
                continue

            # For the initials, override does have an affect due to some people having weird notes in their name.
            # For example:
            #   yang-liu-georgetown's name is Yang (Janet) Liu
            #   For the time being, clean name does not remove the (Janet) from the name (might change later)
            #   So yang-liu-georgetown's initials are [y,j,l]. But we WANT to compare this to the target of yang-liu,
            #   so we override it
            cleaned_initials = [w[0] for w in cleaned_name.split()]
            same_initials = True
            if len(cleaned_initials) != len(target_initials) and not override_with_sim:
                if pass_sim_test:
                    warnings.append(
                        "{} passed the similarity test, but does not have the same number of initials".format(_id))
                    warnings.append("target name ={}".format(target_id))
                    warnings.append("author name ={}".format(name))
                continue
            for i in range(len(target_initials)):
                if target_initials[i] != cleaned_initials[i]:
                    same_initials = False
                    break
            if not same_initials and not override_with_sim:
                if pass_sim_test:
                    warnings.append("{} passed the similarity test, but does not have the same initials".format(_id))
                    warnings.append("target name ={}".format(target_id))
                    warnings.append("author name ={}".format(name))
                continue

            if pass_sim_test:
                if override_with_sim and not same_initials:
                    debug.append("{} was added due to overriding with sim score".format(_id))
                debug.append("{} is similar to {}".format(_id, target_id))
                out.append(_id)
        debug.append("Found {} similar authors".format(len(out)))
        return target_id, out, warnings, debug

    def _makePairs(self, target_info, auth_infos):

        self.logger.debug("Making {} pairs".format(len(auth_infos)))
        pbar = tqdm(total=len(auth_infos), file=sys.stdout)
        out = []
        excluded = []
        target, target_paper_info = target_info
        pid_target, target_id = target.split(" ")
        for key, info in auth_infos:
            auth_pid, auth_id = key.split(" ")
            # If the target author shows up in the same paper as the author you are comparing, it is guarnteed they
            # are not the same author
            if auth_pid == pid_target:
                excluded.append([key, info])
                continue
            out.append([target + " " + key, target_paper_info, info])
            pbar.update()
        pbar.close()
        self.logger.debug("len(excluded)={}".format(len(excluded)))
        self.logger.debug("len(out)={}".format(len(out)))
        return out, excluded

    def _determineCorrectAuthor(self, model_results, evaluate=False):
        percent_same = []
        self.logger.debug("len(model_results)={}".format(len(model_results)))
        for k, values in model_results.items():
            try:
                percent_same.append([k, sum(values) / len(values)])
            except Exception as e:
                self.logger.error("sum(values)={}".format(sum(values)))
                self.logger.error("len(values)={}".format(len(values)))
                self.logger.exception(e)
                raise e
        above_threshold = [x for x in percent_same if x[1] * 100 > self.threshold * 100]
        self.logger.debug("{} are above the threshold".format(len(above_threshold)))

        if len(above_threshold) == 1:
            self.logger.debug("Only 1 other author is above the threshold")
            if evaluate:
                return above_threshold[0][0], percent_same
            return above_threshold[0][0], []
        elif len(above_threshold) == 0:
            self.logger.debug("No authors above threshold")
            return None, percent_same
        else:
            self.logger.debug("{} authors above threshold".format(len(above_threshold)))
            """
            If you want to make your own tie breaker, put it here as another elif statement, and pass that value to 
            tie_breaker when initializing
            """
            if self.tie_breaker == "max":
                sums_above_threshold = [[x[0], sum(model_results[x[0]])] for x in above_threshold]
                if evaluate:
                    return max(sums_above_threshold, key=lambda x: x[0])[0], percent_same
                return max(sums_above_threshold, key=lambda x: x[0])[0], above_threshold
            elif self.tie_breaker == "max_percent":
                if evaluate:
                    return max(above_threshold, key=lambda x: x[0])[0], percent_same
                return max(above_threshold, key=lambda x: x[0])[0], above_threshold
            else:
                self.logger.error("tie_breaker is not a valid tie breaker")
                self.logger.exception(ValueError("{} is not a valid tie breaker".format(self.tie_breaker)))
                raise ValueError("{} is not a valid tie breaker".format(self.tie_breaker))

    def __call__(self, target_authors, override_authors=None, evaluation_mode=False):
        if not override_authors:
            override_authors = {}
        override_authors_len = len(override_authors)
        self.logger.debug("__call__ called with arguments: ")
        self.logger.debug("\tlen(target_authors)={}".format(len(target_authors)))
        self.logger.debug("\tlen(override_authors)={}".format(len(override_authors)))
        printLogToConsole(self.console_log_level, "Starting Disambiguation", logging.INFO)
        self.logger.info("Starting Disambiguation")

        has_authors, needs_authors = self._errorCheckCallArgs(target_authors, override_authors)
        ambiguous_authors_res = self._makeAmbiguousAuthors(has_authors, needs_authors, override_authors)
        ambiguous_papers, ambiguous_names, check_authors, authors_to_get, excluded_authors = ambiguous_authors_res
        self.logger.debug("{} authors had no similar authors".format(len(excluded_authors)))

        ambiguous_papers_to_use = {x: ambiguous_papers[x] for x in ambiguous_papers if x not in excluded_authors}
        to_compare, excluded = self._makeAmbiguousPairs(ambiguous_papers_to_use, check_authors, authors_to_get)

        # initialize it here so that even if not using self.same_paper_diff_people, it can run without any errors
        known_different = {}
        if self.same_paper_diff_people:
            self.logger.debug("Removing excluded")
            to_compare, known_different = self._removeKnownDifferent(to_compare, excluded)

        compare_results = self._compareAmbiguousPairs(to_compare)
        compare_results = self._consolidateResults(compare_results)
        predictions, probabilities = self._makePredictions(compare_results)

        if self.use_probabilities:
            to_use = {}
            for k, info in probabilities.items():
                to_use[k] = {x: [y[1] for y in info[x]] for x in info.keys()}
        else:
            to_use = predictions

        warning_auth = []
        correct_dict = defaultdict(dict)
        printLogToConsole(self.console_log_level, "Determining the correct author", logging.INFO)
        self.logger.info("Determining the correct author")
        pbar = tqdm(total=len(predictions), file=sys.stdout)
        for k, pred in to_use.items():
            self.logger.debug("{}")
            correct, above_thres = self._determineCorrectAuthor(pred, evaluation_mode)
            correct_dict[k]["same"] = correct
            correct_dict[k]["different"] = [x for x in pred.keys() if x != correct]
            self.logger.debug("{} was determined to be the same as {}".format(k, correct))
            if evaluation_mode:
                correct_dict[k]["percent_same"] = above_thres
            if len(above_thres) != 1 and not evaluation_mode:
                self.logger.debug("Added {} to warnings".format(k))
                warning_auth.append([k, above_thres])
            correct_dict[k]["papers_affected"] = ambiguous_papers[k]
            pbar.update()
        pbar.close()
        printLogToConsole(self.console_log_level, "Writing results to results.json", logging.INFO)
        self.logger.info("Writing results to results.json")
        with open("results.json", "w") as f:
            json.dump(correct_dict, f, indent=4, sort_keys=True)

        return correct_dict

    def _errorCheckCallArgs(self, target_authors, override_authors):
        has_authors = []
        needs_authors = []
        self.logger.debug("Checking target_authors and override_authors for errors")
        for i in target_authors:
            if i not in self.author_papers:
                self.logger.exception(KeyError("{} not in author_papers".format(i)))
                raise KeyError("{} not in author_papers".format(i))
            if i not in override_authors:
                self.logger.debug("{} is not in override_authors, adding to need_authors".format(i))
                needs_authors.append(i)
            else:
                self.logger.debug("{} is in override_authors, adding to has_authors".format(i))
                has_authors.append(i)
        for k, info in override_authors.items():
            if k not in target_authors:
                self.logger.exception(ValueError("{} is in override_authors and not in target_authors".format(k)))
                raise ValueError("{} is in override_authors and not in target_authors".format(k))
            if not isinstance(info, list):
                raise ValueError("override_authors expects a dict of lists")
            for a in info:
                if a in target_authors:
                    self.logger.exception(ValueError("{} cannot be compared to itself".format(a)))
                    raise ValueError("{} cannot be compared to itself".format(a))
                if a not in self.author_papers:
                    self.logger.exception(KeyError("{} is not a valid author".format(a)))
                    raise KeyError("{} is not a valid author".format(a))
        return has_authors, needs_authors

    @staticmethod
    def _compareAuthors(args):
        try:
            comparator, target_key, pairs = args
        except ValueError as e:
            raise e

        out = defaultdict(list)
        for pair_key, a, b in pairs:
            p1, a_id, p2, b_id = pair_key.split(" ")
            if p1 + " " + a_id != target_key:
                raise ValueError("Attempting to compare a pair that is not a target")
            compare_results = comparator([pair_key, 0, a, b])
            out[b_id].append(compare_results[-1])
        return target_key, out

    def _makeAmbiguousAuthors(self, has_authors, needs_authors, override_authors):
        ambiguous_author_papers = defaultdict(list)
        ambiguous_author_names = dict()
        authors_get_info = list()
        check_author_keys = defaultdict(list)
        excluded = []
        for i in [*has_authors, *needs_authors]:
            ambiguous_author_papers[i] = self.author_papers.pop(i)
            try:
                ambiguous_author_names[i] = cleanName(nameFromDict(self.id_to_name[i])).lower()
                del self.author_name[i]
            except KeyError as e:
                self.logger.warning("{} is not in id_to_name".format(i))
                excluded.append(i)

        for a in has_authors:
            if a in excluded:
                self.logger.debug("Skipping {} because it is in excluded".format(a))
                continue
            authors_get_info.extend(override_authors[a])
            check_author_keys[a] = self._makeCheckAuthors(override_authors[a])
        args = []
        for a in needs_authors:
            if a in excluded:
                self.logger.debug("Skipping {} because it is in excluded".format(a))
                continue
            args.append([a, ambiguous_author_names[a], self.author_name, self.str_algorithm, self.name_similarity_cutoff,
                         self.sim_overrides])
        printLogToConsole(self.console_log_level, "Getting similar authors in parallel with {} cores".format(self.cores),
                          logging.INFO)
        self.logger.info("Getting similar authors in parallel with {} cores".format(self.cores))
        sim_authors = []
        with mp.Pool(self.cores) as Pool:
            imap_results = list(tqdm(Pool.imap_unordered(self._getSimilarAuthors, args), total=len(args), file=sys.stdout))
            for target, auth, warnings, debug in imap_results:
                sim_authors.append([target, auth])
                for i in warnings:
                    self.logger.warning(i)
                for i in debug:
                    self.logger.debug(i)

        pbar = tqdm(total=len(sim_authors), file=sys.stdout)
        for a, auths in sim_authors:

            if a in override_authors:
                self.logger.exception(ValueError("{} is in need authors, but is already in override_authors".format(a)))
                raise ValueError("{} is in need authors, but is already in override_authors".format(a))
            pbar.write("INFO: Checking similar authors to {}".format(a))
            self.logger.info("Checking similar authors to {}".format(a))
            if len(auths) == 0:
                self.logger.warning("{} has no similar authors".format(a))
                excluded.append(a)
            else:
                authors_get_info.extend(auths)
                check_author_keys[a] = self._makeCheckAuthors(auths)
            pbar.update()
        pbar.close()
        authors_get_info = list(set(authors_get_info))
        return ambiguous_author_papers, ambiguous_author_names, check_author_keys, authors_get_info, excluded

    def _makeCheckAuthors(self, check_author):
        out = []
        for i in check_author:
            try:
                for p in self.author_papers[i]:
                    out.append((p, i))
            except:
                self.logger.warning("{} was not found in self.author_papers".format(i))
        return out

    def _makeAmbiguousPairs(self, ambiguous_papers, check_authors, authors_to_get):
        printLogToConsole(self.console_log_level, "Creating pairs for ambiguous authors", logging.INFO)
        self.logger.info("Creating pairs for ambiguous authors")

        known_author_info, error_authors, error_papers = self._getAuthorInfos(authors_to_get)
        if error_authors > 0:
            self.logger.warning("{} errors getting known author infos".format(error_authors))
        if error_papers > 0:
            self.logger.warning("{} errors getting known author papers".format(error_papers))

        self.logger.debug("{} known papers".format(len(known_author_info)))
        self.logger.debug("{} ambiguous author ids".format(len(check_authors)))
        results = defaultdict(list)
        excluded = defaultdict(list)
        for a in ambiguous_papers.keys():
            printLogToConsole(self.console_log_level, "Creating pairs for {}".format(a), logging.INFO)
            self.logger.info("Creating pairs for {}".format(a))
            self.logger.debug("{} has {} papers".format(a, len(ambiguous_papers[a])))
            self.logger.debug("{} has {} to check against".format(a, len(check_authors[a])))
            self.logger.debug("{} has {} total possible pairs".format(a, len(ambiguous_papers) * len(check_authors[a])))

            known_to_use = [[" ".join(x), known_author_info[" ".join(x)]] for x in check_authors[a]]
            for p in ambiguous_papers[a]:
                ambiguous_paper_info = getAuthorInfo([self.papers[p], a])
                pairs_to_use, pairs_excluded = self._makePairs(ambiguous_paper_info, known_to_use)
                self.logger.debug("{} {} has {} pairs".format(p, a, len(pairs_to_use)))
                self.logger.debug("{} {} has {} excluded".format(p, a, len(pairs_excluded)))
                results[" ".join([p, a])] = pairs_to_use

                excluded[" ".join([p, a])] = [x[0] for x in pairs_excluded]

        return results, excluded

    def _compareAmbiguousPairs(self, pairs_to_use):
        printLogToConsole(self.console_log_level, "Comparing all ambiguous pairs", logging.INFO)
        self.logger.info("Comparing all ambiguous pairs")
        try:
            comparator = CompareAuthors(**self.compare_args)
        except Exception as e:
            self.logger.error("Error intializing comparator")
            self.logger.error("comparator_args={}".format(list(self.compare_args.keys())))
            self.logger.exception(e)
            raise e
        out = {}
        if self.cores == 1:
            self.logger.debug("Using 1 core")
            pbar = tqdm(total=len(pairs_to_use), file=sys.stdout)
            for k, pairs in pairs_to_use.items():
                target, compare_results = self._compareAuthors([comparator, k, pairs])
                out[target] = compare_results
                pbar.update()
            pbar.close()
            return out
        else:
            self.logger.debug("Using {} cores".format(self.cores))
            args = [[comparator, k, pairs] for k, pairs in pairs_to_use.items()]
            with mp.Pool(self.cores) as Pool:
                imap_results = list(
                    tqdm(Pool.imap_unordered(self._compareAuthors, args), total=len(args), file=sys.stdout))
            for k, res in imap_results:
                out[k] = res
            return out

    def _removeKnownDifferent(self, pairs, excluded):
        fixed_pairs = {}

        known_different = {}
        self.logger.debug("Getting all authors who are in excluded")
        for k, info in excluded.items():
            if len(info) == 0:
                continue
            _, k_id = k.split(" ")
            if k_id not in known_different:
                known_different[k_id] = set()
            for a in info:
                _, a_id = a.split(" ")
                known_different[k_id].add(a_id)
        for k in known_different.keys():
            known_different[k] = list(known_different[k])
        self.logger.debug("Removing pairs that have the excluded authors")
        for k, info in pairs.items():
            pid, k_id = k.split(" ")
            if k_id not in known_different:
                fixed_pairs[k] = info
                continue
            fixed_pairs[k] = []
            for pair in info:
                p1, a, p2, b = pair[0].split(' ')
                if p1 != pid or a != k_id:
                    raise ValueError("Pairs[{}] has a pair that has {} as the first value".format(k, " ".join([p1, a])))
                if b not in known_different[k_id]:
                    fixed_pairs[k].append(pair)
                else:
                    self.logger.debug("Removed {} from the pairs b/c it was in known_different".format(pair[0]))
        return fixed_pairs, known_different

    def _consolidateResults(self, compare_results):
        printLogToConsole(self.console_log_level, "Consolidating Compare results", logging.INFO)
        self.logger.info("Consolidating Compare results")
        out = {}
        pbar = tqdm(total=len(compare_results), file=sys.stdout)
        for k, results in compare_results.items():
            pid, k_id = k.split(" ")
            if k_id not in out:
                out[k_id] = defaultdict(list)
            for _id, id_results in results.items():
                out[k_id][_id].extend(id_results)
            pbar.update()
        pbar.close()
        self.logger.debug("Converting to np arrays")
        author_compare_results = {}
        for author, info in out.items():
            author_results = {}
            self.logger.debug("Converting {} results".format(author))
            for other_id, compare_arrays in info.items():
                self.logger.debug("Consolidating results from {}".format(other_id))
                if any([1 for x in compare_arrays if len(x) != self.compare_terms]):
                    self.logger.error(
                        "A compare result from {}-{} does not have the correct number of terms".format(author,
                                                                                                       other_id))
                    self.logger.error("Lengths are: {}".format([len(x) for x in compare_results]))
                    self.logger.error("Expected length is: {}".format(self.compare_terms))
                    raise ValueError("Compare results length does not match comparator's result length")
                try:
                    author_results[other_id] = np.array(compare_arrays)
                except Exception as e:
                    self.logger.warning(
                        "Ran into exception {} when converting compare_results, trying array by array".format(e))
                    tmp_arrays = []
                    for a in compare_results:
                        tmp_arrays.append(np.asarray(a))
                    author_results[other_id] = np.asarray(tmp_arrays).reshape(len(compare_arrays), self.compare_terms)

            author_compare_results[author] = author_results
        return author_compare_results

    def _makePredictions(self, author_arrays):
        printLogToConsole(self.console_log_level, "Predicting same authors", logging.INFO)
        self.logger.info("Predicting same authors")
        predictions = defaultdict(dict)
        probabilities = defaultdict(dict)
        pbar = tqdm(total=len(author_arrays), file=sys.stdout)
        for target, info in author_arrays.items():
            pbar.write("INFO: Predicting same authors to {}".format(target))
            self.logger.info("Predicting same authors to {}".format(target))
            for author, results in info.items():
                self.logger.debug("Making predictions for {}".format(author))
                predictions[target][author] = self.model.predict(results).tolist()
                try:
                    probabilities[target][author] = self.model.predict_proba(results).tolist()
                except:
                    self.logger.warning("Could not get probabilities for {} - {} ".format(target, author))
            pbar.update()
        pbar.close()

        return predictions, probabilities
