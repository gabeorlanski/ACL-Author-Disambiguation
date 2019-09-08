import sys
from collections import defaultdict, Counter
from copy import deepcopy
import logging
from tqdm import tqdm
from src.utility_functions import printLogToConsole, createLogger
import os
from src.paper import Paper


class TargetCreator:
    parameters = dict(
        treat_id_different_people=[False, "When inputting targets, treat every paper associated with the id as an unique person"],
        raise_error=[False, "Raise an exception when error check fails"],
        skip_error_papers=[False,
                           "Skip papers that had an error for one author id, even if the author id in question was not the one that caused the "
                           "paper to be added to error_papers"],
        one_target_per_paper=[False, "Allow only a single target per paper"]
    )

    def __init__(self, papers, id_to_name, author_papers, treat_id_different_people=False,
                 console_log_level=logging.ERROR, file_log_level=logging.DEBUG, log_format=None, log_path=None,
                 raise_error=False, skip_error_papers=False, one_target_per_paper=False, save_data=False, ext_directory=False, save_path=None,
                 cores=4):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/disambiguation.log"
        self.logger = createLogger("disambiguator", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.treat_id_different_people = treat_id_different_people
        self.papers = {}
        for k, p in papers.items():
            if isinstance(p, Paper):
                self.papers = papers
                break
            self.papers[k] = Paper(**p)
        self.id_to_name = deepcopy(id_to_name)
        self.author_papers = deepcopy(author_papers)
        self.author_id_suffix = Counter()
        self.raise_error = raise_error
        self.error_papers = set()
        self.new_papers = {}
        self.new_author_papers = defaultdict(list)
        self.new_id_to_name = {}
        self.old_ids = set()
        self.skip_errors = skip_error_papers
        self.one_per_paper = one_target_per_paper
        self.save_data = save_data
        self.ext_directory = ext_directory
        self.save_path = save_path
        self.cores = cores

    def _updatePapers(self, old_id, new_id, papers=None):
        if old_id not in self.author_papers:
            if self.raise_error:
                raise ValueError("{} is not in author_papers".format(old_id))
            else:
                self.logger.warning("{} is not in author papers".format(old_id))
                return
        if not papers:
            self.logger.debug("No papers passed, using self.author_papers[{}]".format(old_id))
            papers = self.author_papers[old_id]

        self.logger.debug("Adding {} papers".format(len(papers)))
        for p in papers:
            if p not in self.papers[p]:
                self.logger.warning("{} is not parsed, skipping".format(p))
                continue
            paper = self.papers[p]
            pid = paper.pid
            self.logger.debug("Adding {} with new id {}".format(pid, new_id))

            # Check if the old_id is in the paper, if it is not, add it to
            if old_id not in paper.authors or old_id not in paper.affiliations:
                self.logger.debug("Did not find {} in {}".format(old_id, pid))
                if self.raise_error:
                    raise ValueError("{} is not in authors or affiliations".format(old_id))
                self.logger.warning("{} is not in {}'s authors or affiliations, skipping and adding to error_papers".format(old_id, pid))
                self.error_papers.add(pid)
                continue

            if pid in self.error_papers:
                if self.skip_errors:
                    self.logger.warning("Skipping paper {} because it is in error_papers".format(pid))
                    continue
                else:
                    self.logger.debug("{} was in error_papers, but skip_errors is false".format(pid))

            # Check if the paper has already been modified once
            if pid in self.new_papers:
                if self.one_per_paper:
                    self.logger.warning("Skipping {} as it already has a target".format(pid))
                    continue
                else:
                    self.logger.debug("{} already has a target, but will still add the target {}".format(pid, new_id))
                    paper = self.new_papers[pid]

            # Check if the paper has already been modified to use the temporary id
            if new_id in self.new_author_papers:
                if pid in self.new_author_papers[new_id]:
                    self.logger.warning("Attempting to change {} from id {} to {} that was already changed".format(pid, old_id, new_id))
                    if self.raise_error:
                        raise ValueError("{} has already been changed from id {} to {}".format(pid, old_id, new_id))
                    else:
                        self.logger.debug("Skipping paper {}".format(pid))
                        continue

            # Check if the temporary id is already in the paper to modify
            if new_id in paper.authors or new_id in paper.affiliations:
                self.logger.warning("Target id {} is already in paper {} but is not in author_papers".format(new_id, pid))
                if self.raise_error:
                    raise ValueError("{} is already in {}".format(new_id, pid))
                else:
                    self.logger.debug("Adding {} to {}'s papers".format(pid, new_id))
                    if pid not in self.new_papers:
                        self.logger.debug("Adding paper {} to papers".format(pid))
                        self.new_papers[pid] = paper
                    self.new_author_papers[new_id].append(pid)
                    continue

            # Copy to ensure that data is not lost when I delete old_id
            old_auth = deepcopy(paper.authors[old_id])
            old_aff = deepcopy(paper.affiliations[old_id])
            del paper.affiliations[old_id]
            del paper.authors[old_id]
            paper.authors[new_id] = old_auth
            paper.affiliations[new_id] = old_aff

            self.new_papers[pid] = paper
            self.new_author_papers[new_id].append(paper)

    def _handleTarget(self, target, papers):
        self.logger.debug("Handling target {}".format(target))

        self.author_id_suffix[target] += 1
        tmp_id = target + str(self.author_id_suffix[target])
        while tmp_id in self.author_papers:
            self.author_id_suffix[target] += 1
            tmp_id = target + str(self.author_id_suffix[target])
        new_id = tmp_id

        self._updatePapers(target, new_id, papers)
        old_name = self.id_to_name[target]
        if new_id in self.new_id_to_name:
            if old_name != self.new_id_to_name[new_id]:
                self.logger.warning("{} is already in id_to_name, but the names are different".format(new_id))
                self.logger.debug("old_name={}".format(old_name))
                self.logger.debug("new_name={}".format(self.id_to_name[new_id]))
                if self.raise_error:
                    raise ValueError("{}'s name in id_to_name is not the same as {}".format(new_id, target))
                else:
                    self.logger.debug("Updating new name to reflect old_id")
                    self.id_to_name[new_id] = old_name
        else:
            self.new_id_to_name[new_id] = old_name
        self.old_ids.add(target)
        return new_id

    def createTarget(self, user_target, papers=None):
        if papers is None:
            if self.treat_id_different_people:
                rtr_ids = []
                for p in self.author_papers[user_target]:
                    self.logger.debug("handling paper {}".format(p))
                    new_id = self._handleTarget(user_target, [p], None)
                    rtr_ids.append(new_id)
                    self.logger.debug("new_id={}".format(new_id))
                return rtr_ids
            else:
                return [self._handleTarget(user_target, self.author_papers[user_target])]

        else:
            return [self._handleTarget(user_target, papers)]

    def fillData(self):
        printLogToConsole(self.console_log_level, "Adding rest of data to new author papers", logging.INFO)
        self.logger.info("Adding rest of data to new author papers")
        auth_pbar = tqdm(total=len(self.author_papers), file=sys.stdout)
        for a in self.author_papers.keys():
            if a in self.new_author_papers:
                self.logger.debug("Skipping author {}, in new_author_papers".format(a))
            elif a in self.old_ids:
                self.logger.debug("Skipping author {}, in old_ids".format(a))
            else:
                if a not in self.id_to_name:
                    self.logger.warning("{} is in author_papers but not in id_to_name".format(a))
                else:
                    self.new_author_papers[a] = deepcopy(self.author_papers[a])
                    self.new_id_to_name[a] = self.id_to_name[a]
            auth_pbar.update()
        auth_pbar.close()
        printLogToConsole(self.console_log_level, "Adding papers", logging.INFO, logger=self.logger)
        paper_pbar = tqdm(total=len(self.papers), file=sys.stdout)
        for pid, paper in self.papers.items():
            if pid in self.new_papers:
                self.logger.debug("{} is already in self.new_papers".format(pid))
            elif pid in self.error_papers:
                self.logger.debug("{} is in error_papers, but not in self.new_papers".format(pid))
            else:
                self.new_papers[pid] = paper
            paper_pbar.update()
        paper_pbar.close()

        return self.new_papers, self.new_author_papers, self.new_id_to_name
