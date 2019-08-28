import os
import lxml
from io import StringIO
import yaml
import json
import re
from lxml import etree
from collections import defaultdict, Counter
import fuzzysearch
import unidecode
from html import unescape
from tqdm import tqdm
import logging
from copy import deepcopy
from textdistance import JaroWinkler
from src.utility_functions import *
from src.paper import Paper
from multiprocessing import Pool
import sys

remove_html = re.compile("<[^>]*>")
remove_punct = re.compile("[^\w\s]")
remove_punct_ids = re.compile("[^\w\s-]")

# Saves time to define them once, and they are never at risk of being overwritten
xpath_config = json.load(open("config.json"))["PDFParserXpaths"]
namespaces = xpath_config["namespaces"]
get_authors = etree.XPath(xpath_config["get_authors"], namespaces=namespaces)
get_name = etree.XPath(xpath_config["get_name"], namespaces=namespaces)
get_author_affiliation = etree.XPath(xpath_config["get_author_affiliation"], namespaces=namespaces)
get_author_email = etree.XPath(xpath_config["get_author_email"], namespaces=namespaces)
check_department = etree.XPath(xpath_config["check_department"], namespaces=namespaces)
get_abstract = etree.XPath(xpath_config["get_abstract"], namespaces=namespaces)
get_affiliations = etree.XPath(xpath_config["get_affiliations"], namespaces=namespaces)
get_orgs = etree.XPath(xpath_config["get_orgs"], namespaces=namespaces)
get_address = etree.XPath(xpath_config["get_address"], namespaces=namespaces)
jaro_winkler = JaroWinkler()


class PDFParser:
    def __init__(self, parsed_path, save_data=False, save_dir="/data",
                 ext_directory=False, similarity_cutoff=.75, manual_fixes=None, print_errors=False):
        # XPaths for parsing, doing it here rather than inside of the function because it is runs faster

        self.parsed_path = parsed_path

        self.aliases = {}
        self.papers = {}
        self.id_to_name = {}
        self.final_papers = {}
        self.organizations = {}
        self.same_names = {}
        self.parsed = {}
        self.save_data = save_data
        self.save_dir = save_dir
        self.ext_directory = ext_directory
        self.similarity_cutoff = similarity_cutoff
        self.print_errors = print_errors
        self.incomplete_papers = []
        if not manual_fixes:
            self.manual_fixes = {}

    def loadData(self, papers, aliases, id_to_name, same_names,manual_fixes):
        self.papers = {x: Paper(**papers[x]) for x in papers.keys()}
        self.aliases = deepcopy(aliases)
        self.id_to_name = deepcopy(id_to_name)
        self.same_names = deepcopy(same_names)
        self.manual_fixes = deepcopy(manual_fixes)

    def _dataErrorCheck(self):
        if not self.papers:
            raise ValueError("self.papers is None")
        elif not isinstance(self.papers, dict):
            raise TypeError("self.papers must be a dict")
        if not self.aliases:
            raise ValueError("self.aliases is None")
        elif not isinstance(self.aliases, dict):
            raise TypeError("self.aliases must be a dict")
        if not self.id_to_name:
            raise ValueError("self.id_to_name is None")
        elif not isinstance(self.id_to_name, dict):
            raise TypeError("self.id_to_name must be a dict")
        if not self.same_names:
            raise ValueError("self.same_names is None")
        elif not isinstance(self.same_names, list):
            raise TypeError("self.same_names must be a list")

    def getOrganizations(self, r):
        affiliations = get_affiliations(r)
        out = {}
        for a in affiliations:

            # Getting organization info
            org_names = get_orgs(a)
            org_information = defaultdict(list)
            for o in org_names:
                org_text = o.text
                if "type" not in o.attrib:
                    continue
                if org_text not in org_information[o.attrib["type"]]:
                    org_information[o.attrib["type"]].append(org_text)

            type_count = 0
            tmp_name = None
            tmp_type = None
            type_to_name = {}
            if "department" not in org_information:
                org_information["department"] = []
            for k in org_information.keys():
                if k == "department":
                    continue
                if type_count > 1:
                    tmp_name.append(org_information[k][0])
                    tmp_type.append(k)
                else:
                    tmp_name = [org_information[k][0]]
                    tmp_type = [k]
                    type_to_name[k] = org_information[k][0]

            if tmp_name:
                # org_id = createID(fullname=)
                org_type = list(set(tmp_type))
                if len(tmp_name) == 1:
                    org_id = createID(fullname=tmp_name[0])
                else:
                    if "institution" in type_to_name:
                        org_id = createID(fullname=type_to_name["institution"])
                    elif "laboratory" in type_to_name:
                        org_id = createID(fullname=type_to_name["laboratory"])
                    else:
                        org_id = None

            else:
                org_id = None
                org_information["name"] = []
                org_type = []

            # Getting address info. Because there can possibly be more than one namespace defined, I put in this quick
            # safeguard to guarantee the tags do not show up in the data. This WILL break if for some reason there are multiple
            # namespace tags on different address elements. But I could not see way in which that could occur, so faster to
            # handle it like this, then have a lot of appends and checking dicts and lists for something existing in them
            address = {}
            address_results = get_address(a)
            for k in namespaces.keys():
                if any([x.tag for x in address_results if namespaces[k] in x.tag]):
                    # Had to put in the second replace because the {} were remaining
                    address = {x.tag.replace(str(namespaces[k]), "").replace("{}", ""): x.text for x in address_results}
                    break
            try:
                out[a.attrib["key"]] = {
                    "address": address,
                    "info": org_information,
                    "id": org_id,
                    "type": org_type
                }
            except KeyError as e:
                continue
        return out

    def _matchAuthors(self, actual, found):
        remaining_found = deepcopy(found)
        remaining_actual = deepcopy(actual)
        out = {}
        unknown = []
        for tmp_id, name_f in found:
            for j, name_a in actual:
                if name_f == name_a:
                    out[(tmp_id, name_f)] = j
                    remaining_found.remove((tmp_id, name_f))
                    remaining_actual.remove((j, name_a))
                    break
        for tmp_id, name_f in remaining_found:
            sim_score = []
            for j, name_a in remaining_actual:
                sim_score.append(((j, name_a), jaro_winkler.similarity(name_f, name_a)))
            if sim_score:
                best_match = max(sim_score, key=lambda x: x[1])
            else:
                best_match = ["NONE_FOUND", 0]
            if best_match[1] >= self.similarity_cutoff:
                out[(tmp_id, name_f)] = best_match[0][0]
                remaining_actual.remove(best_match[0])
            else:
                unknown.append((tmp_id, name_f))
        return out, unknown

    def _parseAuthors(self, actual, r, pid, manual_fixes):
        out = {}
        authors_found = get_authors(r)
        unknown = []
        errors = []
        correct_with_manual = 0
        tmp_author_info = {}
        found_info = []
        found_count = Counter()
        manual_fixes_required = []
        keys_with_same = []
        for person in authors_found:
            try:
                name = getChildText(get_name(person)[0], delimiter=" ").replace("  ", " ")
            except:
                errors.append("WARNING:{}: issue with getting a name from an author in paper".format(pid))
                continue
            _id = createID(fullname=name)
            corresponding_actual = _id
            name_to_use = cleanName(name.lower())
            in_same = False
            if name in self.same_names or any([x for x in self.same_names if name.lower() == x.lower()]):
                in_same = True
            else:
                if _id in self.id_to_name:
                    name_to_use = cleanName(nameFromDict(self.id_to_name[_id]).lower())
                elif _id not in actual:
                    alias = None
                    junk_removed = unidecode.unidecode(unescape(name.lower()))
                    punct_removed = remove_punct_ids.sub("", junk_removed)
                    if name.lower() in self.aliases:
                        alias = name.lower()
                    elif junk_removed in self.aliases:
                        alias = junk_removed
                    elif punct_removed in self.aliases:
                        alias = punct_removed
                    # I do this separately because some people have - in their aliases
                    elif punct_removed.replace("-", " ") in self.aliases:
                        alias = punct_removed.replace("-", " ")

                    if alias:
                        corresponding_actual = self.aliases[alias]
                        name_to_use = cleanName(alias)
                    else:
                        corresponding_actual = _id
            try:
                aff_key = get_author_affiliation(person)[0]
            except:
                aff_key = None
            try:
                author_email = get_author_email(person)[0]
            except:
                author_email = None

            if (corresponding_actual, aff_key, author_email) in manual_fixes:
                correct_with_manual += 1
                fixed_id = manual_fixes[(corresponding_actual, aff_key, author_email)]
                out[fixed_id] = {
                    "email": author_email,
                    "aff_key": aff_key
                }
            elif in_same:
                keys_with_same.append(((corresponding_actual, name_to_use), {
                    "email": author_email,
                    "aff_key": aff_key
                }))
            else:
                found_info.append(((corresponding_actual, name_to_use), {
                    "email": author_email,
                    "aff_key": aff_key
                }))
                found_count[(corresponding_actual, name_to_use)] += 1
        if len(keys_with_same) == 1:
            unknown.append(keys_with_same[0][0])
            tmp_author_info[keys_with_same[0][0]] = keys_with_same[0][1]
        else:
            for k, i in keys_with_same:
                manual_fixes_required.append((k, i))
        for tmp_key, info in found_info:
            if found_count[tmp_key] > 1:
                manual_fixes_required.append((tmp_key, info))
            else:
                if tmp_key[0] in actual:
                    out[tmp_key[0]] = info
                else:
                    tmp_author_info[tmp_key] = info
                    unknown.append(tmp_key)
        remaining = [(x, cleanName(actual[x].lower())) for x in actual.keys() if x not in out]
        fixed, unknown = self._matchAuthors(remaining, unknown)
        fixed_count = 0
        for k in fixed.keys():
            fixed_count += 1
            out[fixed[k]] = tmp_author_info[k]
        # for k in unknown:
        #     manual_fixes_required.append((k,tmp_author_info[k]))

        return out, manual_fixes_required, unknown, fixed_count, correct_with_manual, errors

    def _parsePaper(self, p, pdf_xml, manual_fixes):

        # In theory all papers should be copied before being passed to this function, this is just a safeguard to
        # guarantee no synchronization problems if papers are being parsed using multiprocessing
        out = p.copy()
        true_authors = out.authors
        unknown_authors = []
        warnings = []
        root = etree.XML(pdf_xml)
        try:
            affs = self.getOrganizations(root)
        except:
            return None, -1, ["ERROR:{}: No affiliations found for paper".format(out.pid)]

        if not out.abstract:
            try:
                out.abstract = str(get_abstract(root)[0])
            except:
                return None, -1, ["ERROR:{}: No abstract found for paper".format(out.pid)]

        auth_result = self._parseAuthors(true_authors, root, out.pid, manual_fixes)
        parsed_authors, manual_fixes_required, unknown_authors, fixed_count, correct_with_manual, errors = auth_result
        warnings.extend(errors)
        affiliations = {}
        for author, info in parsed_authors.items():
            aff = info["aff_key"]
            if not aff:
                affiliations[author] = {
                    "email": info["email"],
                    "affiliation": {
                        "id": None,
                        "type": [],
                        "info": None
                    }
                }
            else:
                if aff not in affs:
                    warnings.append("WARNING:{}: An affiliation was found that was not parsed. Author: {}".format(out.pid,
                                                                                                                  author))
                    affiliations[author] = {
                        "email": info["email"],
                        "affiliation": {
                            "id": None,
                            "type": {},
                            "info": None
                        }
                    }
                else:
                    affiliations[author] = {
                        "email": info["email"],
                        "affiliation": affs[aff]
                    }
        out.unknown = deepcopy(unknown_authors)
        out.affiliations = deepcopy(affiliations)
        return [out, manual_fixes_required, fixed_count, correct_with_manual], 0, warnings

    def _paperParseWrapper(self, args):
        return self._parsePaper(*args)

    def __call__(self, xml_path, debug_cutoff=None):

        self._dataErrorCheck()
        if xml_path[-1] != '/':
            xml_path = xml_path + '/'
        parsed_pdfs = [f for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and ".xml" in f]
        if debug_cutoff:
            parsed_pdfs = parsed_pdfs[:debug_cutoff]
        errors = []
        errors_pre_parse = 0
        papers_before_parse = 0
        warnings = []
        manual_fixes_needed = {}
        total_correct_manual = 0
        total_fixed = 0
        print("INFO: Parsing PDFs")
        args_pbar = tqdm(total=len(parsed_pdfs), file=sys.stdout, dynamic_ncols=True)
        for f in parsed_pdfs:
            try:
                pid = f.split(".")[0]
            except IndexError as e:
                args_pbar.update()
                errors.append("ERROR:{}: failed to get the pid".format(f))
                errors_pre_parse += 1
                continue
            current_paper = self.papers[pid].copy()
            try:
                with open(xml_path + f, "rb") as fb:
                    try:
                        root = fb.read()
                    except:
                        args_pbar.update()
                        errors_pre_parse += 1
                        errors.append("ERROR:{}: PDF's xml could not be parsed by lxml".format(pid))
                        continue
            except:
                args_pbar.update()
                errors_pre_parse += 1
                errors.append("ERROR:{}: PDF's xml could not be read by lxml".format(pid))
                continue

            if pid in self.manual_fixes:
                man_fixes = self.manual_fixes[pid]
            else:
                man_fixes = {}
            papers_before_parse += 1
            rtr, rc, errors = self._parsePaper(current_paper, root, man_fixes)
            if rc >= 0:
                paper, manual_fixes, fixed, correct_manual = rtr
                warnings.extend(errors)
                self.parsed[paper.pid] = paper
                total_correct_manual += correct_manual
                total_fixed += fixed
                if manual_fixes:
                    manual_fixes_needed[paper.pid] = manual_fixes
            else:
                if self.print_errors:
                    for e in errors:
                        args_pbar.write(e)
                errors.extend(errors)
            args_pbar.update()

        args_pbar.close()
        results = [
            ["Total Papers Parsed", len(self.parsed)],
            ["Failures", papers_before_parse - len(self.parsed)],
            ["Manual Fixes Needed", len(manual_fixes_needed)],
            ["Total Fixes", total_fixed],
            ["Corrected with manual fixes", total_correct_manual],
            ["Warnings", len(warnings)]
        ]
        printStats("PDF Parsing Stats", results)
        print("INFO: Generating POS for titles")
        with tqdm(file=sys.stdout,total=len(self.parsed)) as pbar:
            for k in self.parsed.keys():
                self.parsed[k].createPOS()
                pbar.update()
            pbar.close()
        for k in manual_fixes_needed.keys():
            self.incomplete_papers.append(k)
        if self.save_data:
            json_path = self.save_dir
            csv_path = self.save_dir
            txt_path = self.save_dir

            if self.ext_directory:
                json_path = json_path + "/json"
                csv_path = csv_path + "/csv"
                txt_path= txt_path + "/txt"
                if not os.path.exists(json_path):
                    os.mkdir(json_path)
                if not os.path.exists(csv_path):
                    os.mkdir(csv_path)
                if not os.path.exists(txt_path):
                    os.mkdir(txt_path)
            print("INFO: Writing parsed papers")
            write_papers_pbar = tqdm(total=len(self.parsed), file=sys.stdout)
            papers_print = {}
            for x in self.parsed.keys():
                papers_print[x] = self.parsed[x].asDict()
                write_papers_pbar.update()
            write_papers_pbar.close()
            with open(json_path + "/parsed_papers.json", "w") as f:
                json.dump(papers_print, f, indent=4)

            print("INFO: Writing manual fixes needed")
            write_fixes_pbar = tqdm(total=len(manual_fixes_needed), file=sys.stdout)
            with open(csv_path + "/manual_fixes_needed.csv", "w") as f:
                for k in manual_fixes_needed.keys():
                    for i in manual_fixes_needed[k]:
                        auth_name, info = i
                        email = info["email"] if info["email"] else "None"
                        aff_key = info["aff_key"] if info["aff_key"] else "None"

                        f.write("{},{},{},{},{}".format(k, auth_name[0], auth_name[1], email, aff_key))
                        f.write("\n")
                    write_fixes_pbar.update()
            write_fixes_pbar.close()
            with open(txt_path+ "/incomplete_papers.txt", "w") as f:
                for p in self.incomplete_papers:
                    f.write(p+"\n")

            if json_path != csv_path:
                print("INFO: Wrote json files to {}".format(json_path))
                print("INFO: Wrote csv files to {}".format(csv_path))
                print("INFO: Wrote txt files to {}".format(txt_path))
            else:
                print("INFO: Wrote ACL files to ".format(csv_path))
