from src.utility_functions import createLogger, createID, cleanName, printLogToConsole, nameFromDict
import logging
import os
import json
import re
from collections import defaultdict

remove_weird_notes = re.compile("\(\w+\)")


class InputHandler:
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
            if k == "yang-liu-georgetown":
                print("DEBUG")
            name_cleaned =cleanName(remove_weird_notes.sub(" ", nameFromDict(name)).replace("  ", " ")).replace("  "," ")
            self.names[name_cleaned].append(k)
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

        self.valid_main_commands = {
            "t": {
                "required": "target-id",
                "desc": "Specify target",
                "action": self._addTarget
            },
            "d": {
                "required": None,
                "desc": "Display a target or all targets",
                "action": self._displayTargets
            },
            "r": {
                "required": None,
                "desc": "Remove a target",
                "action": self._removeTarget

            },
            "g": {
                "required": "target-id",
                "desc": "Generate authors to compare with based on their name",
                "action": self._genAuthorOverride

            },
            "c": {
                "required": "target-id",
                "desc": "Clear target's authors to compare with",
                "action": self._clearAuthorOverride

            },
            "o": {
                "required": None,
                "desc": "Display override authors",
                "action": self._displayOverride
            },
            "h": {
                "required": None,
                "desc": "Help",
                "action": self._printHelp
            },
            "s": {
                "required": None,
                "desc": "Save targets",
                "action": self._save
            },
            "e": {
                "required": None,
                "desc": "Finish and continue",
                "action": None
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

    def handleUserInput(self):
        printLogToConsole(self.console_log_level, "Found {} targets".format(len(self.targets)), logging.INFO, logger=self.logger)
        print("INFO: Enter target-ids:")
        while 1:
            user_input = input(">> ")
            if user_input == "\n" or len(user_input) == 0:
                self.logger.debug("Got empty user input")

            command_split = user_input.split(" ")
            command = command_split[0]
            if command not in self.valid_main_commands:
                printLogToConsole(self.console_log_level, "{} is not a valid command".format(command), logging.INFO, logger=self.logger)
            elif command == "e":
                if self.confirmAction():
                    return
            else:
                action = self.valid_main_commands[command]["action"]
                if self.valid_main_commands[command]["required"]:
                    if len(command_split) != 2:
                        print("Invalid arguments passed to {}".format(command))
                    else:
                        action(command_split[1])
                else:
                    action()
                self.logger.debug("Finished running command {}".format(command))

    def _validAuthor(self, a):
        self.logger.debug("Checking if {} is a valid author-id".format(a))
        if a not in self.author_papers:
            self.logger.debug("{} is not in author_papers".format(a))
            return -1
        at_least_one_parsed = False
        for p in self.author_papers[a]:
            self.logger.debug("Checking paper {}".format(p))
            if p in self.papers:
                at_least_one_parsed = True
        if not at_least_one_parsed:
            self.logger.debug("{} has no parsed papers".format(a))
            return -2
        return 0

    def _getAuthorInfo(self, a):
        self.logger.debug("Getting info for {}".format(a))
        name = cleanName(remove_weird_notes.sub(" ", nameFromDict(self.id_to_name[a])).replace("  ", " ")).replace("  "," ")
        printLogToConsole(self.console_log_level, "id={}".format(a), logging.INFO, logger=self.logger)
        printLogToConsole(self.console_log_level, "name={}".format(name), logging.INFO, logger=self.logger)
        printLogToConsole(self.console_log_level, "Papers for {}:".format(a), logging.INFO, logger=self.logger)
        for p in self.author_papers[a]:
            if p not in self.papers:
                continue
            try:
                title = self.papers[p].title
            except:
                title = self.papers[p]["title"]
            printLogToConsole(self.console_log_level, "\t{}\t{}".format(p, title), logging.INFO, logger=self.logger)
        printLogToConsole(self.console_log_level, "{} Author(s) have this name".format(len(self.names[name])), logging.INFO, logger=self.logger)

    @staticmethod
    def confirmAction():
        while 1:
            confirmation = input("Confirm Action [Y/N]?: ")
            if confirmation.lower() == "y":
                return True
            elif confirmation.lower() == "n":
                return False

    def _addTarget(self, a):
        self.logger.debug("Received add command with arguments {}".format(a))
        valid_author = self._validAuthor(a)
        if valid_author == -1:
            printLogToConsole(self.console_log_level, "{} is not a valid author id".format(a), logging.INFO, logger=self.logger)
        elif valid_author == -2:
            printLogToConsole(self.console_log_level, "{} has no parsed papers".format(a), logging.INFO, logger=self.logger)
        else:
            self._getAuthorInfo(a)
            self.targets.append(a)

            if a not in self.override_authors:
                printLogToConsole(self.console_log_level, "{} does not have specified authors to compare with".format(a), logging.INFO,
                                  logger=self.logger)
            else:
                printLogToConsole(self.console_log_level, "{} has {} authors to compare with".format(a,len(self.override_authors[a])), logging.INFO,
                                  logger=self.logger)
            printLogToConsole(self.console_log_level, "{} added to targets".format(a), logging.INFO, logger=self.logger)
            printLogToConsole(self.console_log_level, "{} current targets".format(len(self.targets)), logging.INFO, logger=self.logger)

        return

    def _removeTarget(self):
        self.logger.debug("Received remove command")
        print("INFO: Select the number of the id you would like to remove from targets, enter e to exit")
        if len(self.targets) == 0:
            printLogToConsole(self.console_log_level, "No possible targets to remove", logging.INFO, logger=self.logger)
            return
        while 1:
            for i, v in enumerate(self.targets):
                print("INFO: [{}] {}".format(i, v))
            to_remove = input(">>")
            if to_remove == "e":
                self.logger.debug("Exit command received")
                return
            try:
                to_remove = int(to_remove)
            except ValueError:
                printLogToConsole(self.console_log_level, "{} is not valid".format(to_remove), logging.INFO, logger=self.logger)
                continue
            if to_remove < 0:
                printLogToConsole(self.console_log_level, "{} is not valid".format(to_remove), logging.INFO, logger=self.logger)
                continue
            elif to_remove >= len(self.targets):
                printLogToConsole(self.console_log_level, "{} is not valid".format(to_remove), logging.INFO, logger=self.logger)
                continue
            else:
                printLogToConsole(self.console_log_level, "{} is selected to be removed".format(self.targets[to_remove]), logging.INFO, logger=self.logger)
                if self.confirmAction():
                    self.targets.remove(self.targets[to_remove])
                    printLogToConsole(self.console_log_level, "Remaining targets:", logging.INFO,
                                      logger=self.logger)
                    for i, v in enumerate(self.targets):
                        print("INFO: {}".format(v))
                    return
                else:
                    self.logger.debug("User did not confirm action")

    def _printHelp(self):
        self.logger.debug("Received help command")
        print("INFO: Commands available:")
        for k, v in self.valid_main_commands.items():
            if v["required"]:
                print("\t{} {:<10} {}".format(k, v["required"], v["desc"]))
            else:
                print("\t{:<12} {}".format(k, v["desc"]))

    def _save(self):
        self.logger.debug("Received save command")
        return

    def _displayTargets(self):
        self.logger.debug("Received display command")
        for a in self.targets:
            if a in self.override_authors:
                print("INFO: {:<20} {} papers and {} authors specified to compare with".format(a,len(self.author_papers[a]),len(self.override_authors[a])))
            else:
                print("INFO: {:<20} {} papers and {} authors specified to compare with".format(a,len(self.author_papers[a]),0))
        return

    def _clearAuthorOverride(self, a):
        self.logger.debug("Received clear override command".format(a))
        if a not in self.override_authors:
            printLogToConsole(self.console_log_level, "{} does not have authors to compare with".format(a), logging.INFO, logger=self.logger)
            return
        if self.confirmAction():
            del self.override_authors[a]
            self.logger.debug("Removed {} from override_authors".format(a))

    def _genAuthorOverride(self,a):
        self.logger.debug("Received clear override command".format(a))
        if a in self.override_authors:
            printLogToConsole(self.console_log_level, "{} already has authors to compare with".format(a), logging.INFO, logger=self.logger)
            return
        elif a not in self.targets:
            printLogToConsole(self.console_log_level, "{} is not a target".format(a), logging.INFO, logger=self.logger)
            return
        name = cleanName(remove_weird_notes.sub(" ", nameFromDict(self.id_to_name[a])).replace("  ", " ")).replace("  "," ")
        print("INFO: Other authors with the same name:")
        for other_a in self.names[name]:
            if other_a !=a:
                print("INFO: {}".format(other_a))
        if len(self.names[name]) == 1:
            printLogToConsole(self.console_log_level, "{} only has {}, will not add authors to compare with".format(name, a), logging.INFO, logger=self.logger)
        else:
            self.override_authors[a] = [x for x in self.names[name] if x != a]
            self.logger.debug("{} authors added to override_authors".format(len(self.names[name])))
            
    def _displayOverride(self):
        printLogToConsole(self.console_log_level, "Override Authors: ", logging.INFO,logger=self.logger)
        for k in self.override_authors.keys():
            printLogToConsole(self.console_log_level, "{} has {} authors to compare with".format(k, len(self.override_authors[k])), logging.INFO,
                              logger=self.logger)