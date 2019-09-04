from lxml import etree
from io import StringIO
import json
import os
import re
from collections import defaultdict, Counter
import unidecode
from html import unescape
import time
from tqdm import tqdm
import numpy as np
from src.utility_functions import printStats, nameFromDict, createID, printLogToConsole, getChildText, \
    remove_punct_ids, \
    convertPaperToSortable, createLogger
import yaml
import logging
import sys
from src.paper import Paper


class ACLParser:
    parameters = dict(
        existing_data=[False, "Load already Parsed data. Doesn't override them"],
    )

    def __init__(self, xpath_config=None, save_data=False, save_path="/data", ext_directory=False, existing_data=False,
                 file_log_level=logging.DEBUG, console_log_level=logging.ERROR, log_format=None, log_path=None,
                 cores=4):
        """
        :param xpath_config: Dictionary of xpaths to use, You MUST pass this
        :type xpath_config: dict(str=str)
        :param save_data: Save data for later use, defaults to False
        :type save_data: bool
        :param ext_directory: Save data into directories based on their file type. Is ignored if save_data is False.
        defaults to False.
        :type ext_directory: bool
        :param save_path: Directory to save data, Is ignored if save_data is False.  defaults to None
        :type save_path: str
        :param existing_data: Load existing data, defaults to False
        :type existing_data: bool
        :param file_log_level: logging level to file (default is debug)
        :type file_log_level: logging.level
        :param console_log_level: logging level to console (default is ERROR)
        :type console_log_level: logging.level
        :param log_format: format of log messages (defaults to '%(asctime)s|%(levelname)8s|%(module)20s|%(
        funcName)20s: %(message)s')
        :type log_format: str
        :param log_path: path to log files(default is '/logs/acl_parser.log')
        :type log_path: str
        :param cores: Doesn't Do anything, just put it in to avoid errors with how I handle configs
        :type cores: int
        """
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/acl_parser.log"
        self.logger = createLogger("create_training_data", log_path, log_format, console_log_level,
                                   file_log_level)
        self.console_log_level = console_log_level
        if xpath_config is None:
            raise ValueError("no xpath_config was passed")
        self.aliases = {}
        self.same_name = []
        self.similar_names = {}
        self.id_to_name = {}
        self.affiliations = {}
        self.papers = {}
        self.conflicts = defaultdict(list)
        self.parser = etree.HTMLParser(encoding="utf-8")
        self.use_existing_data = existing_data
        self.cores = cores
        # Xpaths used for parsing the paper xml
        self.get_volumes = etree.XPath(xpath_config["volumes"])
        self.get_papers = etree.XPath(xpath_config["papers"])
        self.get_title = etree.XPath(xpath_config["titles"])
        self.get_authors = etree.XPath(xpath_config["authors"])
        self.get_abstract = etree.XPath(xpath_config["abstracts"])
        self.get_pid = etree.XPath(xpath_config["pids"])
        self.get_first_name = etree.XPath(xpath_config["first_name"])
        self.get_last_name = etree.XPath(xpath_config["last_name"])
        self.check_volume = etree.XPath(xpath_config["check_volume"])

        self.save_data = save_data
        if save_path[-1] == "/":
            save_path = save_path[:-1]
        self.save_dir = save_path
        self.ext_directory = ext_directory

    def __call__(self, xml_path, variant_path):
        """
        Run the parser
        :param xml_path: the path to a directory of xml files
        :type xml_path: str
        :param variant_path: path to the variants file
        :type variant_path: str
        """
        self.parseNameVariants(variant_path)
        self.parseACLXml(xml_path)

        if self.save_data:
            json_path = self.save_dir
            txt_path = self.save_dir

            if self.ext_directory:
                json_path = json_path + "/json"
                txt_path = txt_path + "/txt"
                if not os.path.exists(json_path):
                    os.mkdir(json_path)
                if not os.path.exists(txt_path):
                    os.mkdir(txt_path)

            with open(json_path + "/aliases.json", "w") as f:
                json.dump(self.aliases, f, indent=4)
            with open(json_path + "/id_to_name.json", "w") as f:
                json.dump(self.id_to_name, f, indent=4)

            papers_print = {x: self.papers[x].asDict() for x in self.papers.keys()}
            with open(json_path + "/acl_papers.json", "w") as f:
                json.dump(papers_print, f, indent=4)

            with open(json_path + "/conflicts.json", "w") as f:
                json.dump(self.conflicts, f, indent=4)
            with open(json_path + "/known_affiliations.json", "w") as f:
                json.dump(self.affiliations, f, indent=4)
            with open(json_path + "/similar_names.json", "w") as f:
                json.dump(self.similar_names, f, indent=4)
            with open(txt_path + "/same_names.txt", "w") as f:
                for i in self.same_name:
                    f.write(i + "\n")
            if json_path != txt_path:
                printLogToConsole(self.console_log_level, "Wrote json files to {}".format(json_path), logging.INFO,
                                  logger=self.logger)
                printLogToConsole(self.console_log_level, "Wrote txt files to {}".format(txt_path), logging.INFO,
                                  logger=self.logger)
            else:
                printLogToConsole(self.console_log_level, "Wrote ACL files to {}".format(txt_path), logging.INFO,
                                  logger=self.logger)

    def parseNameVariants(self, variant_path):
        """
        Parse the name variants from the file
        :param variant_path: Path to the name variant file. At the moment this must NOT include the actual name of
        the file as that is hardcoded to be name_variants.yaml
        :type variant_path: str
        """
        # TODO: Implement argument to specify name of name_variants file
        raw_aliases = yaml.load(open(variant_path + "/name_variants.yaml").read(), Loader=yaml.FullLoader)
        pbar = tqdm(total=len(raw_aliases), file=sys.stdout, dynamic_ncols=True, ascii=" =")
        printLogToConsole(self.console_log_level, "Parsing name_variant.yaml", logging.INFO, print_func=pbar.write,
                          logger=self.logger)
        self.logger.debug("{} raw aliases".format(len(raw_aliases)))
        for p in raw_aliases:
            first = p["canonical"]["first"]
            last = p["canonical"]["last"]
            name = nameFromDict(p["canonical"])
            if "id" in p:
                key = p["id"]
            else:
                key = createID(first, last)
            if "comment" in p:
                if "several people" in p["comment"].lower():
                    self.logger.debug("{}(key={}) had 'several people' in their comment".format(name, key))
                    self.same_name.append(name)
                else:
                    self.logger.debug("{}(key={}) had an affiliation in their comment".format(name, key))
                    self.affiliations[key] = p["comment"]
            if "similar" in p:
                self.logger.debug("{}(key={}) had similar in their name".format(name, key))
                self.similar_names[key] = p["similar"]
            if "variants" in p:
                for variant in p["variants"]:
                    if variant["first"] and variant["last"]:
                        self.aliases[nameFromDict(variant).lower().strip()] = key
                        self.aliases[nameFromDict(variant).lower().strip().replace(".", "")] = key
                    elif variant["last"]:
                        self.aliases[variant["last"].lower()] = key
                    else:
                        self.aliases[variant["first"].lower()] = key

            self.id_to_name[key] = p["canonical"]
            self.logger.debug("Added the id {} with the name {} to id_to_name".format(key, name))
            pbar.update()
        pbar.close()
        self.same_name = list(set(self.same_name))

    def parseACLXml(self, xml_path):
        """
        Parse the ACL XMLs to find papers. Modifies:\n
        - id_to_name\n
        - papers\n
        - conflicts\n
        :param xml_path: path to the ACL
        :type xml_path: str
        """
        xml_files = [f for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and ".xml" in f]
        ids_to_create = 0
        found_ids = 0
        aliases_corrected = 0
        total_papers = 0
        papers_failed = 0
        author_count = []
        people_no_id = defaultdict(list)
        pbar = tqdm(total=len(xml_files), file=sys.stdout, dynamic_ncols=True, ascii=" =")
        printLogToConsole(self.console_log_level, "Parsing ACL xml files", logging.INFO, print_func=pbar.write,
                          logger=self.logger)
        self.logger.debug("{} xml files to parse".format(len(xml_files)))
        for f in xml_files:

            self.logger.debug("Parsing {}".format(f))
            parsed = []
            with open(xml_path + f, "rb") as fb:
                root = etree.XML(fb.read())
            if root is None:
                printLogToConsole(self.console_log_level, "{} could not be read".format(f), logging.WARNING,
                                  print_func=pbar.write,
                                  logger=self.logger)

            for v in self.get_volumes(root):
                parsed.extend(self.get_papers(v))
            pre_papers_failed = papers_failed
            for p in parsed:
                total_papers += 1
                rtr, status, msg = self._parsePaper(p)
                if status == 0:
                    paper, p_found, a_found, no_ids = rtr
                    self.papers[paper.pid] = paper
                    found_ids += len(p_found)
                    for _id, name in p_found:
                        if _id not in self.id_to_name:
                            self.id_to_name[_id] = name
                    for name, paper in no_ids:
                        people_no_id[name].append(paper)
                    aliases_corrected += a_found
                    ids_to_create += len(no_ids)
                    author_count.append(len(p_found) + a_found + len(no_ids))
                else:
                    # pbar.write("WARNING: A paper in {} failed to parse with message {}".format(f,msg))
                    self.logger.warning("A paper in {} failed to parse with message {}".format(f, msg))
                    papers_failed += 1
            if papers_failed > pre_papers_failed:
                printLogToConsole(self.console_log_level,
                                  "{} papers in {} had an issue".format(papers_failed - pre_papers_failed, f),
                                  logging.WARNING, pbar.write,
                                  self.logger)
            pbar.update()
        pbar.close()

        self.conflicts, resolved = self._createNewID(people_no_id)
        results = [
            ["Total Papers Parsed", total_papers],
            ["Successes", total_papers - papers_failed],
            ["Failures", papers_failed],
            ["Total ids", len(self.id_to_name)],
            ["IDs created", resolved],
            ["ID Conflicts", len(self.conflicts)],
            ["Aliases Corrected", aliases_corrected],
            ["With abstracts", len([x for x in self.papers.keys() if self.papers[x].abstract])],
            ["Average Authors", float(np.mean(np.asarray(author_count)))],

        ]
        printStats("Results", results, line_adaptive=True)

    def _parsePaper(self, paper):
        """
        Parse a paper
        :param paper: The paper to parse
        :type paper: lxml.etree._Element
        :return: The parsed paper, new ids, aliases found, people with no ids, return status, error messages
        :rtype: [Paper(), list(str), int, list(str)],int, list(str)
        """
        title = getChildText(self.get_title(paper)[0])
        authors = {}
        new_ids = []
        no_ids = []
        aliases_found = 0
        try:
            pid = self.get_pid(paper)[0]
        except IndexError as e:
            return [None, None, None], -1, "no pid found"
        if "https://www.aclweb.org" in pid:
            pid = pid.split("/")[-1]
        elif "http://" in pid:
            return [None, None, None], -1, "http:// in the paper id"
        elif "https://" in pid:
            return [None, None, None], -1, "https:// in the paper id"
        try:
            abstract = self.get_abstract(paper)[0]
        except IndexError as e:
            abstract = None
        for a in self.get_authors(paper):

            first_name = None
            last_name = None
            try:
                first_name = self.get_first_name(a)[0]
                last_name = self.get_last_name(a)[0]
                name = first_name + " " + last_name
            except IndexError:
                if self.get_first_name(a):

                    name = self.get_first_name(a)[0]
                    first_name = name
                elif self.get_last_name(a):
                    name = self.get_last_name(a)[0]
                    last_name = name
                else:
                    return [Paper(pid, title=title, abstract=abstract, authors=authors), new_ids,
                            aliases_found, no_ids], -1, "ERROR: Author does not have either a first or last name"

            # It didn't seem to catch without this
            if not first_name and not last_name:
                return [Paper(pid, title=title, abstract=abstract, authors=authors), new_ids,
                        aliases_found, no_ids], -1, "ERROR: Author does not have either a first or last name"

            if "id" in a.attrib:
                authors[a.attrib["id"]] = name

                if a.attrib["id"] not in self.id_to_name:
                    new_ids.append((a.attrib["id"], name))
            else:
                alias_found = False
                alias = name.lower()
                leave_dashes = remove_punct_ids.sub("", name.lower())

                if name.lower() in self.aliases:
                    alias_found = True
                elif leave_dashes in self.aliases:
                    alias_found = True
                    alias = leave_dashes
                elif leave_dashes.replace("-", " ") in self.aliases:
                    alias_found = True
                    alias = leave_dashes.replace("-", " ")
                if not alias_found:
                    no_ids.append(((first_name, last_name), pid))
                else:
                    authors[self.aliases[alias]] = name
                    aliases_found += 1

        return [Paper(pid, title=title, abstract=abstract, authors=authors), new_ids,
                aliases_found, no_ids], 0, None

    def _createNewID(self, people_no_id):
        """
        Go through and create new ids and modify papers in self.papers to reflect the new ids found. If there are
        possible conflicts, it will add them to self.conflicts. and give them an integer at the end of their id. This
        is determined by the order of their earliest paper.
        :param people_no_id: dict of people with no id and their papers
        :type people_no_id: {str:list(str)}
        :return: the ids with conflicts and all temporary ids found that had the conflict ids, and a count of
        resolved ids
        :rtype: collections.defaultdict(list), int
        """
        id_to_people = defaultdict(list)
        conflicts = []
        conflict_ids = defaultdict(list)
        resolved = 0

        for person in people_no_id.keys():
            try:
                _id = createID(*person)
            except AttributeError as e:
                print(person)
                break
            id_to_people[_id].append(person)

        for _id, name in id_to_people.items():
            if len(name) > 1:
                conflicts.append((_id, {
                    "first": name[0][0],
                    "last": name[0][1]
                }))
                continue
            resolved += 1
            self.id_to_name[_id] = {
                "first": name[0][0],
                "last": name[0][1]
            }
            try:
                str_name = name[0][0] + " " + name[0][1]
            except:
                if not name[0][1]:
                    str_name = name[0][0]
                else:
                    str_name = name[0][1]
            for p in people_no_id[name[0]]:
                self.papers[p].authors[_id] = str_name
        self.logger.debug("Found {} possible conflicts".format(len(conflicts)))
        for conflict, name in conflicts:
            earliest_paper = []
            for p in id_to_people[conflict]:
                lowest_author = min([convertPaperToSortable(x) for x in people_no_id[p]])
                earliest_paper.append((p, lowest_author))
            people_count = 0
            earliest_paper = sorted(earliest_paper, key=lambda x: x[1])
            for a in earliest_paper:
                key = conflict if people_count == 0 else conflict + str(people_count)
                for p in people_no_id[a[0]]:
                    self.papers[p].authors[key] = nameFromDict(name)
                self.id_to_name[key] = name
                conflict_ids[conflict].append((key, " ".join(a[0])))
                people_count += 1

        return conflict_ids, resolved
