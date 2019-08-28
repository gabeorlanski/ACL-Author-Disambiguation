import sys
from collections import defaultdict, Counter
from copy import deepcopy
import logging
from tqdm import tqdm
from src.utility_functions import printLogToConsole, createLogger
import os
from src.paper import Paper


class DisambiguationInputHandler:
    def __init__(self, papers, id_to_name, author_papers, treat_id_different_people=False,
                 console_log_level=logging.ERROR, file_log_level=logging.DEBUG, log_format=None, log_path=None,
                 raise_error_check_remove=False):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/disambiguation.log"
        self.logger = createLogger("disambiguator", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.treat_id_different_people = treat_id_different_people
        self.papers = {}
        for k, p in papers.items():
            self.papers[k] = Paper(**p)
        self.id_to_name = deepcopy(id_to_name)
        self.author_papers = deepcopy(author_papers)
        self.author_id_suffix = Counter()
        self.raise_error_check_remove = raise_error_check_remove

    def _updatePapers(self, old_id, new_id, papers):
        if not papers:
            papers = self.author_papers[old_id]
        if new_id in self.author_papers:
            raise ValueError("{} is already in self.author_papers".format(new_id))
        if new_id in self.id_to_name:
            raise ValueError("{} is already in self.id_to_name")
        self.id_to_name[new_id] = deepcopy(self.id_to_name[old_id])
        self.author_papers[new_id] = papers
        paper_changed_count = 0
        for p in papers:
            if p not in self.author_papers[old_id]:
                raise KeyError("{} is not in old_id papers".format(p))
            self.author_papers[old_id].remove(p)
            if new_id in self.papers[p].authors or new_id in self.papers[p].affiliations:
                raise KeyError("{} is already in paper {}".format(new_id, p))
            old_auth_name = self.papers[p].authors.pop(old_id)
            old_aff = self.papers[p].affiliations.pop(old_id)

            self.papers[p].authors[new_id] = old_auth_name
            self.papers[p].affiliations[new_id] = old_aff
            paper_changed_count +=1
        self.logger.debug("{} papers changed from {} to {}".format(paper_changed_count,old_id,new_id))
        if len(self.author_papers[old_id]) == 0:
            printLogToConsole(self.console_log_level,"Checking if {} is same to remove".format(old_id),logging.INFO)
            self.logger.info("Checking if {} is same to remove".format(old_id))
            self._checkSafeRemove(old_id)

    def _checkSafeRemove(self, _id):
        can_remove = True
        self.logger.debug("Checking if it is safe to remove {}".format(_id))
        pbar = tqdm(total=len(self.papers), file=sys.stdout)
        for p in self.papers:
            if _id in self.papers[p].affiliations or self.papers[_id].authors:
                can_remove = False
                error_str = "{} is still in paper {}, but has no papers associated with him".format(_id, p)
                if self.raise_error_check_remove:
                    raise ValueError(error_str)
                else:
                    self.logger.warning(error_str)
                    self.author_papers[_id].append(p)
                pbar.update()
        pbar.close()
        if can_remove:
            removed_name = self.id_to_name.pop(_id)
            removed_papers = self.author_papers.pop(_id)
            if len(removed_papers) > 0:
                raise ValueError("removed_papers is not empty")
            self.logger.debug("removed_name={}".format(removed_name))

    def _handleTarget(self, target, papers, override_id):
        self.logger.debug("Handling target {}".format(target))
        if override_id is None:
            self.author_id_suffix[target] += 1
            tmp_id = target+str(self.author_id_suffix[target])
            while tmp_id in self.author_papers:
                self.author_id_suffix[target] +=1
                tmp_id = target + str(self.author_id_suffix[target])
            new_id = tmp_id
        else:
            new_id = override_id

        self._updatePapers(target,new_id,papers)

        return new_id

    def handleInput(self,user_target,papers=None,override_id=None):
        if papers is None:
            if self.treat_id_different_people:
                rtr_ids = []
                if override_id is not None:
                    self.logger.warning("treat_id_different_people is not False, ignoring override_id")
                for p in self.author_papers[user_target]:
                    self.logger.debug("handling paper {}".format(p))
                    new_id = self._handleTarget(user_target,[p],None)
                    rtr_ids.append(new_id)
                    self.logger.debug("new_id={}".format(new_id))
                return rtr_ids
            else:
                return self._handleTarget(user_target, papers, override_id)

        else:
            return self._handleTarget(user_target,papers,override_id)
