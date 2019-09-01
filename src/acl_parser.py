"""
Written by Gabriel Orlanski
"""
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
from src.utility_functions import *
import yaml
import logging
import sys
from src.paper import Paper
import warnings


class ACLParser:
    def __init__(self, xpath_config=None, save_data=False, save_path="/data", ext_directory=False, existing_data=None,file_log_level=logging.DEBUG,
                 console_log_level=logging.ERROR, log_format=None, log_path=None, cores=4):
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
        self.save_dir = save_path
        self.ext_directory = ext_directory

    def __call__(self, xml_path, variant_path):
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

            with open(json_path+"/aliases.json", "w") as f:
                json.dump(self.aliases, f, indent=4)
            with open(json_path+ "/id_to_name.json", "w") as f:
                json.dump(self.id_to_name, f, indent=4)

            papers_print = {x: self.papers[x].asDict() for x in self.papers.keys()}
            with open(json_path+"/acl_papers.json", "w") as f:
                json.dump(papers_print, f, indent=4)

            with open(json_path+"/conflicts.json", "w") as f:
                json.dump(self.conflicts, f, indent=4)
            with open(json_path+"/known_affiliations.json", "w") as f:
                json.dump(self.affiliations, f, indent=4)
            with open(json_path+"/similar_names.json", "w") as f:
                json.dump(self.similar_names, f, indent=4)
            with open(txt_path+"/same_names.txt", "w") as f:
                for i in self.same_name:
                    f.write(i + "\n")
            if json_path != txt_path:
                print("INFO: Wrote json files to {}".format(json_path))
                print("INFO: Wrote txt files to {}".format(txt_path))
            else:
                print("INFO: Wrote ACL files to ".format(txt_path))

    def parseNameVariants(self, variant_path):
        raw_aliases = yaml.load(open(variant_path+ "/name_variants.yaml").read(), Loader=yaml.FullLoader)
        pbar = tqdm(total=len(raw_aliases), file=sys.stdout, dynamic_ncols=True, ascii=" =")
        pbar.write("Parsing name_variants.yaml...")
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
                    self.same_name.append(name)
                else:
                    self.affiliations[key] = p["comment"]
            if "similar" in p:
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
            pbar.update()
        pbar.close()
        self.same_name = list(set(self.same_name))

    def parseACLXml(self, xml_path):
        xml_files = [f for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and ".xml" in f]
        ids_to_create = 0
        found_ids = 0
        aliases_corrected = 0
        total_papers = 0
        papers_failed = 0
        author_count = []
        people_no_id = defaultdict(list)
        pbar = tqdm(total=len(xml_files), file=sys.stdout, dynamic_ncols=True, ascii=" =")
        pbar.write("Parsing xml files...")
        for f in xml_files:
            parsed = []
            root = None
            with open(xml_path + f, "rb") as fb:
                root = etree.XML(fb.read())
            if root is None:
                raise pbar.write("{} could not be read".format(f))
            for v in self.get_volumes(root):
                parsed.extend(self.get_papers(v))

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
                    papers_failed += 1
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
                conflict_ids[conflict].append((key," ".join(a[0])))
                people_count += 1

        return conflict_ids, resolved
