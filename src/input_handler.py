from src.utility_functions import createLogger, createID, cleanName, printLogToConsole, nameFromDict
import logging
import os
import json
import re
from collections import defaultdict

remove_weird_notes = re.compile("\(\w+\)")


class InputHandler:
    parameters
    def __init__(self, papers, author_papers, id_to_name, console_log_level=logging.ERROR, file_log_level=logging.DEBUG, log_format=None,
                 log_path=None, target_path=None, save_data=False, ext_directory=False, save_path=None, cores=4):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/disambiguation.log"
        self.logger = createLogger("input_handler", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.papers = papers
        self.author_papers = author_papers
        self.id_to_name = id_to_name
        self.names = defaultdict(list)
        for k, name in id_to_name.items():
            self.names[cleanName(remove_weird_notes.sub(" ",name).replace("  "," "))].append(k)
        self.save_data = save_data
        self.save_path = save_path
        self.ext_directory = ext_directory
        self.override_authors = {}
        if not target_path:
            self.logger.debug("No path was passed for target_path")
            self.targets = []
        else:
            self.logger.debug("Opening {}".format(target_path))
            if target_path.split(".")[-1] == "json":
                self.logger.debug("Parsing json...")
                try:
                    targets_dict = json.load(open(target_path))
                except FileNotFoundError as e:
                    self.logger.debug("File path was not found, trying to open with adding os.getcwd()")
                    target_path = "/" + target_path if target_path[0] == "/" else target_path
                    targets_dict = json.load(open(os.getcwd() + target_path))
                for k, v in targets_dict.items():
                    self.logger.debug("Found target {}".format(k))
                    self.targets.append(k)
                    self.override_authors[k] = v
            elif target_path.split(".")[-1] == "txt":
                self.logger.debug("Parsing txt file...")
                try:
                    self.targets = [x.strip() for x in open(target_path).readlines()]
                except FileNotFoundError as e:
                    self.logger.debug("File path was not found, trying to open with adding os.getcwd()")
                    target_path = "/" + target_path if target_path[0] == "/" else target_path
                    self.targets = [x.strip() for x in open(os.getcwd() + target_path).readlines()]
            else:
                self.logger.error("File type {} is not supported".format(target_path.split(".")[-1]))
                raise ValueError("File type {} is not supported".format(target_path.split(".")[-1]))

            self.logger.debug("Found {} targets".format(len(self.targets)))
            self.logger.debug("Found {} overrides".format(len(self.override_authors)))

        # commands in the format {command:{ required value, description}
        self.valid_main_commands = {
            "t": {
                "required": "target-id",
                "desc": "Specify target"
            },
            "m": {
                "required": "target-id",
                "desc": "Modify a target"
            },
            "d": {
                "required": None,
                "optional": "target-id",
                "desc": "Display a target or all targets"
            },
            "r": {
                "required": "target-id",
                "optional": None,
                "desc": "Remove a target"
            },
            "v": {
                "required": "paper-id",
                "optional": None,
                "desc": "View a paper"
            },
            "h": {
                "required": None,
                "desc": "Help"
            },
            "s": {
                "required": None,
                "desc": "Save targets"
            },
            "e": {
                "required": None,
                "desc": "Finish and continue"
            },
        }
        self.valid_target_commands = {
            "a": {
                "required": "target-id",
                "optional": None,
                "desc": "Specify author to compare target to"
            },
            "g": {
                "required": "Author Name",
                "optional": None,
                "desc": "Generate a list of authors to compare to by their name"
            },
            "d": {
                "required": None,
                "optional": None,
                "desc": "Display list of authors"
            },
            "r": {
                "required": "author-id",
                "optional": None,
                "desc": "Remove an author-id"
            },
            "e": {
                "required": None,
                "optional": None,
                "desc": "Finish editing target"
            }

        }

    def handleUserInput(self, user_input):
        if user_input == "\n" or len(user_input) == 0:
            self.logger.debug("Got empty user input")
            return

        command_split = user_input.split(" ")
        command = command_split[0]
        if command not in self.valid_main_commands:
            printLogToConsole(self.console_log_level,"{} is not a valid command".format(command),logging.INFO,logger=self.logger)
            return
    
    def _validAuthor(self,a):
        self.logger.debug("Checking if {} is a valid author-id".format(a))
        if a not in self.author_papers:
            self.logger.debug("{} is not in author_papers")
            return -1
        at_least_one_parsed = False
        for p in self.author_papers:
            if p in self.papers:
                at_least_one_parsed = True
        if not at_least_one_parsed:
            self.logger.debug("{} has no parsed papers")
            return -2
        return 0

    def _getAuthorInfo(self, a):
        self.logger.debug("Getting info for {}".format(a))
        printLogToConsole(self.console_log_level,"id={}".format(a),logging.INFO,logger=self.logger)
        printLogToConsole(self.console_log_level,"name={}".format(nameFromDict(self.id_to_name[a])),logging.INFO,logger=self.logger)
        printLogToConsole(self.console_log_level,"Papers=",logging.INFO,logger=self.logger)
        for p in self.author_papers[a]:
            if p not in self.papers:
                continue
            try:
                title = self.papers[p].title
            except:
                title = self.papers[p]["title"]
            printLogToConsole(self.console_log_level, "--{}\t{}".format(p,title), logging.INFO, logger=self.logger)

    @staticmethod
    def confirmAction():
        while 1:
            confirmation = input("Confirm Action [Y/N]?: ")
            if confirmation.lower() == "y":
                return True
            elif confirmation.lower() == "n":
                return False
