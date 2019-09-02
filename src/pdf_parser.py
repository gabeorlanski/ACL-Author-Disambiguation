import os
from io import StringIO
import yaml
import json
import re
from lxml import etree
from collections import defaultdict, Counter
import fuzzysearch
import ujson
import unidecode
from html import unescape
from tqdm import tqdm
import logging
from copy import deepcopy
from textdistance import JaroWinkler
from py_stringmatching.similarity_measure import soft_tfidf, jaro_winkler
from py_stringmatching.tokenizer import whitespace_tokenizer
from src.utility_functions import cleanName, nameFromDict, createID, printLogToConsole, printStats, chunks, \
    getChildText, createLogger
from src.paper import Paper
import multiprocessing as mp
import sys
from nltk import word_tokenize
import time

remove_html = re.compile("<[^>]*>")
remove_punct = re.compile("[^\w\s]")
remove_punct_ids = re.compile("[^\w\s-]")
parse_section_num = re.compile("(\d+)")
split_address = re.compile("(?<!\w)(\.)|\s?[^\w\s\.]")
# Saves time to define them once, and they are never at risk of being overwritten
try:
    xpath_config = json.load(open("config.json"))["PDFParserXpaths"]
except FileNotFoundError:
    xpath_config = json.load(open("/".join(os.getcwd().split("/")[:-1]) + "/config.json"))
    xpath_config = xpath_config["PDFParserXpaths"]
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
get_citations = etree.XPath(xpath_config["get_citations"], namespaces=namespaces)
get_citation_analytic = etree.XPath(xpath_config["get_citation_analytic"], namespaces=namespaces)
get_citation_title = etree.XPath(xpath_config["get_citation_title"], namespaces=namespaces)
get_citation_authors = etree.XPath(xpath_config["get_citation_authors"], namespaces=namespaces)
get_citation_publication = etree.XPath(xpath_config["get_citation_publication"], namespaces=namespaces)
get_citation_pub_title = etree.XPath(xpath_config["get_citation_pub_title"], namespaces=namespaces)
get_citation_pub_data = etree.XPath(xpath_config["get_citation_pub_data"], namespaces=namespaces)
get_biblScope = etree.XPath(xpath_config["get_biblScope"], namespaces=namespaces)
get_citation_pub_date = etree.XPath(xpath_config["get_citation_pub_date"], namespaces=namespaces)
get_sections = etree.XPath(xpath_config["get_sections"], namespaces=namespaces)
txt_distance_jaro_winkler = JaroWinkler()


class PDFParser:
    def __init__(self, aliases, id_to_name, same_names, sim_cutoff, raise_error=False):
        """
        PDF Parser, parses XML Output of GROBID
        :param aliases: the dictionary of aliases. Key is alias, value is id it relates to
        :param id_to_name: dictionary of ids to their name. key is id, value is id
        :param same_names: list of names that were marked as same names
        :param sim_cutoff: Similarity cutoff for the best match when matching author names to known authors of a paper
        :param raise_error: raise an error instead of return it
        """
        self.aliases = aliases
        self.id_to_name = id_to_name
        self.same_names = same_names
        self.similarity_cutoff = sim_cutoff
        self.raise_error = raise_error

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
            # safeguard to guarantee the tags do not show up in the data. This WILL break if for some reason there
            # are multiple
            # namespace tags on different address elements. But I could not see way in which that could occur,
            # so faster to
            # handle it like this, then have a lot of appends and checking dicts and lists for something existing in
            # them
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
                sim_score.append(((j, name_a), txt_distance_jaro_winkler.similarity(name_f, name_a)))
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
            except Exception as e:
                if self.raise_error:
                    raise e

                errors.append("{} issue with getting a name from an author in paper".format(pid))
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

    def __call__(self, args):
        # Do this so mp.pool.imap_unordered can actually send all the args, otherwise for some reason it wont
        p, pdf_xml, manual_fixes = args
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
            return None, -1, ["No affiliations found for {}".format(out.pid)]

        if not out.abstract:
            try:
                out.abstract = str(get_abstract(root)[0])
            except:
                return None, -1, ["No abstract found for {}".format(out.pid)]

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
                    warnings.append(
                        "WARNING:{}: An affiliation was found that was not parsed. Author: {}".format(out.pid, author))
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

        citations, errors = self._parseCitations(out.pid, root)
        out.citations = citations
        errors.extend(errors)

        sections, status = self._parseSections(root)
        out.sections = sections
        if status != 0:
            if status == -1:
                warnings.append("No sections found in {}".format(out.pid))
            elif status == -2:
                warnings.append("A section failed to parse for {}".format(out.pid))
        out.unknown = deepcopy(unknown_authors)
        out.affiliations = deepcopy(affiliations)
        return [out, manual_fixes_required, fixed_count, correct_with_manual], 0, warnings

    @staticmethod
    def _parseSections(root):
        out = {}
        sections = get_sections(root)
        if not sections:
            return out, -1
        for section in sections:
            try:
                section_number = section.attrib["n"]
                section_title = cleanName(section.text)
            except:
                return out - 2

            section_split = parse_section_num.findall(section_number)
            if section_split[0] not in out:
                out[section_split[0]] = {}
            if len(section_split) == 1:
                out[section_split[0]]["title"] = section_title
            else:
                out[section_split[0]][".".join(section_split[1:])] = section_title

        return out, 0

    @staticmethod
    def _parseCitations(pid, root):
        citations = get_citations(root)
        out = []
        errors = []
        had_error_analytic = False
        had_error_publication = False
        if not citations:
            return out, ["No citations found for paper {}".format(pid)]
        for item in citations:
            try:
                analytic = get_citation_analytic(item)[0]
            except IndexError as e:
                if not had_error_analytic:
                    had_error_analytic = True
                    errors.append("A citation in {} failed get analytic".format(pid))
                continue

            try:
                publication = get_citation_publication(item)[0]
            except IndexError as e:
                if not had_error_publication:
                    errors.append("A citation in {} failed to get publication".format(pid))
                continue

            citation_info = {}
            try:
                citation_title = get_citation_title(analytic)[0]
                citation_info["title"] = cleanName(citation_title.text)
                if "level" in citation_title.attrib:
                    citation_info["type"] = citation_title.attrib["level"]
                else:
                    citation_info["type"] = None
            except IndexError as e:
                citation_info["title"] = None
                citation_info["type"] = None

            authors = get_citation_authors(analytic)
            if not authors:
                citation_info["authors"] = []
            else:
                citation_info["authors"] = [cleanName(getChildText(x, " ").replace("  ", " ")) for x in authors]

            pub_title = get_citation_pub_title(publication)
            if not pub_title:
                citation_info["pub_type"] = None
                citation_info["pub_title"] = None
            else:
                pub_title = pub_title[0]
                if "level" in pub_title.attrib:
                    citation_info["pub_type"] = pub_title.attrib["level"]
                else:
                    citation_info["pub_type"] = None
                citation_info["pub_title"] = cleanName(pub_title.text)

            pub_data = get_citation_pub_data(publication)
            citation_info["volume"] = None
            citation_info["issue"] = None
            citation_info["date"] = None
            citation_info["date_type"] = None
            if pub_data:
                pub_data = pub_data[0]
                biblScopes = get_biblScope(pub_data)
                for i in biblScopes:
                    if "unit" not in i.attrib:
                        continue
                    unit = i.attrib["unit"]
                    if "volume" == unit:
                        try:
                            citation_info["volume"] = int(i.text)
                        except ValueError:
                            citation_info["volume"] = i.text
                    elif "issue" == unit:
                        try:
                            citation_info["issue"] = int(i.text)
                        except ValueError:
                            citation_info["issue"] = i.text
                pub_date = get_citation_pub_date(pub_data)
                if pub_date:
                    pub_date = pub_date[0]
                    if "type" in pub_date.attrib:
                        citation_info["date_type"] = pub_date.attrib["type"]
                    if "when" in pub_date.attrib:
                        try:
                            citation_info["date"] = int(pub_date.attrib["when"].split("-")[0])
                        except IndexError as e:
                            citation_info["date"] = int(pub_date.attrib["when"])
            out.append(citation_info)

        return out, errors

    def parseBatch(self, batches):
        out = []
        for i in batches:
            out.append(self(i))
        return out


class PDFParserWrapper:
    # The main reason I did this was to have an easy way to generate command line arguments with are parse,
    # and maybe for later saving said parameters
    parameters = dict(
        load_parsed=[False, "Use existing parsed_papers.json"],
        allow_load_parsed_errors=[True, "If an error should be thrown if loading parsed failed, Mainly for debugging"],
        assign_similarity_cutoff=[.75, "cutoff for how similar a name must be to assign it to the author name"],
        print_errors=[False, "Print errors to console"],
        parse_parallel_cutoff=[1000, "minimum number of papers needed to run in parallel, ignored if cores=1"],
        parse_batch_size=[200, "Batch size for parallel processing"],
        guess_email_and_aff=[False, "Try to guess emails and affiliations based on frequency."],
        guess_min=[.5, "minimum number of occurrences to guess."],
        combine_orgs=[False, "Combine orgs that are possibly the same."],
        combine_orgs_cutoff=[.8, "Minimum similarity between two orgs to combine them."],
        use_org_most_common=[True, "Use the most common address for an organization"],
        known_affiliations=[False, "Use known affiliations from name_variants.yaml"],
        attempt_fix_parser_errors=[False,
                                   "Attempt to remove possible parser errors by looking if parts of the parsed data "
                                   "appear in other entries. Items that would be affected are organization, email, "
                                   "department."]
    )

    def __init__(self, papers=None, aliases=None, id_to_name=None, same_names=None, manual_fixes=None, load_parsed=False,
                 allow_load_parsed_errors=True, save_data=False, save_path="/data", ext_directory=False,
                 assign_similarity_cutoff=.75, print_errors=False, file_log_level=logging.DEBUG,
                 console_log_level=logging.ERROR, log_format=None, log_path=None, cores=4, parse_parallel_cutoff=1000,
                 parse_batch_size=200, guess_email_and_aff=False, guess_min=.5, combine_orgs=False, combine_orgs_cutoff=.8,
                 use_org_most_common=True, known_affiliations=False, attempt_fix_parser_errors=False):
        """
        Wrapper for the PDF Parser, allows parallel pdf parsing at the expense of memory
        :param papers: Dict of Paper objets or dicts
        :param aliases: dict of aliases where key is alias, value is corresponding id
        :param id_to_name: dict of ids where key is id and value is name
        :param same_names: list of names that have been marked as different people who have the same name
        :param load_parsed: Use existing parsed_papers.json
        :param allow_load_parsed_errors: If an error should be thrown if loading parsed failed, Mainly for debugging
        :param save_data: Save data to disk
        :param save_path: directory where you want data saved
        :param ext_directory: Save each file extension in their own directory (ex create a json directory in the
        save_data path)
        :param assign_similarity_cutoff: cutoff for how similar a name must be to assign it to the author name
        :param manual_fixes: manual fixes you want to make to the parsed paper. Check manual_fixes_needed.csv
        :param print_errors: Print errors to console
        :param file_log_level: Logging level of file
        :param console_log_level: Logging level of console
        :param log_format: Log message format
        :param log_path: path to log file
        :param cores: Number of cores/processes to use
        :param parse_parallel_cutoff: minimum number of papers needed to run in parallel, ignored if cores=1
        :param parse_batch_size: Batch size for parallel processing
        :param guess_email_and_aff: Try to guess emails and affiliations based on frequency. NOT IMPLEMENTED YET
        :param guess_min: minimum number of occurrences to guess. NOT IMPLEMENTED YET
        :param combine_orgs: Combine orgs that are possibly the same. NOT IMPLEMENTED YET
        :param combine_orgs_cutoff: Minimum similarity between two orgs to combine them. NOT IMPLEMENTED YET
        :param use_org_most_common: Use the most common address for an organization
        :param known_affiliations: Use known affiliations from name_variants.yaml NOT IMPLEMENTED YET
        :param attempt_fix_parser_errors: Attempt to remove possible parser errors by looking if parts of the parsed
        data appear in other entries. Items that would be affected are organization, email, department. NOT
        IMPLEMENTED YET
        """
        self.save_data = save_data
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/pdf_parser.log"
        self.console_log_level = console_log_level
        self.logger = createLogger("pdf_parser", log_path, log_format, console_log_level,
                                   file_log_level)
        self.logger.debug("{} aliases".format(len(aliases)))
        self.logger.debug("{} papers".format(len(papers)))
        self.logger.debug("{} ids".format(len(id_to_name)))
        if manual_fixes is not None:
            self.logger.debug("{} manual_fixes".format(len(manual_fixes)))
        else:
            self.logger.debug("0 manual_fixes")

        if not isinstance(aliases, dict):
            self.logger.error("aliases is {}".format(type(aliases)))
            raise ValueError("aliases must be a dict")
        self.aliases = deepcopy(aliases)

        if not isinstance(papers, dict):
            self.logger.error("papers is {}".format(type(papers)))
            raise ValueError("papers must be a dict")
        if isinstance(papers[list(papers.keys())[0]], Paper):
            self.papers = {x: papers[x].copy() for x in papers.keys()}
        else:
            self.papers = {x: Paper(**papers[x]) for x in papers.keys()}

        if not isinstance(id_to_name, dict):
            self.logger.error("id_to_name is {}".format(type(id_to_name)))
            raise ValueError("id_to_name must be a dict")
        self.id_to_name = deepcopy(id_to_name)

        if manual_fixes is not None and not isinstance(manual_fixes, dict):
            self.logger.error("manual_fixes is {}".format(type(manual_fixes)))
            raise ValueError("manual_fixes must be a dict")
        if not manual_fixes:
            self.manual_fixes = {}
        else:
            self.manual_fixes = deepcopy(manual_fixes)

        if same_names is None or not isinstance(same_names, list):
            self.logger.error("same_names is {}".format(type(same_names)))
            raise ValueError("same_names must be a list")
        self.same_names = deepcopy(same_names)

        self.save_data = save_data
        self.save_dir = save_path
        self.ext_directory = ext_directory
        self.org_names = []
        self.department_names = []
        self.organizations = {}
        self.possible_affiliations = {}
        self.author_papers = defaultdict(list)
        self.incomplete_papers = []
        self.effective_org_info = {}
        self.parsed = {}
        if load_parsed:
            try:
                tmp_parsed_path = os.getcwd()
                if save_path is None:
                    tmp_parsed_path = tmp_parsed_path + "/data"
                else:
                    tmp_parsed_path = tmp_parsed_path + save_path
                tmp_txt_path = deepcopy(tmp_parsed_path)
                if self.ext_directory:
                    tmp_txt_path = tmp_txt_path + "/txt/"
                    tmp_parsed_path = tmp_parsed_path + "/json"
                parsed = ujson.load(open(tmp_parsed_path + "/parsed_papers.json"))
                self.parsed = {x: Paper(**parsed[x]) for x in parsed.keys()}
                try:
                    tmp_organizations = json.load(open(tmp_parsed_path + "organizations.json"))
                    for k, info in tmp_organizations.items():
                        tmp_info = {}
                        for s, v in info.items():
                            if s != "count":
                                tmp_info[s] = Counter(v)
                            else:
                                tmp_info[s] = int(v)
                        self.organizations[k] = tmp_info
                    self.effective_org_info = json.load(open(tmp_parsed_path + "effective_org_info.json"))
                except FileNotFoundError:
                    self.organizations = {}
                    self.effective_org_info = {}
                self.department_names = [x.strip() for x in open(tmp_txt_path + "department_corpus.txt").readlines()]
                self.org_names = [x.strip() for x in open(tmp_txt_path + "org_corpus.txt").readlines()]
                self.incomplete_papers = [x.strip() for x in open(tmp_txt_path + "incomplete_papers.txt").readlines()]
            except Exception as e:
                self.logger.warning("load_parsed was passed, but could not load the parsed_papers.json")
                self.logger.exception(e)
                self.parsed = {}
                self.organizations = {}
                self.effective_org_info = {}
                self.org_names = []
                self.department_names = []
                self.incomplete_papers = []
                if not allow_load_parsed_errors:
                    raise e

        self.similarity_cutoff = assign_similarity_cutoff
        self.print_errors = print_errors
        self.cores = cores
        self.parse_parallel_cutoff = parse_parallel_cutoff
        self.batch_size = parse_batch_size
        self.guess_email_and_aff = guess_email_and_aff
        self.guess_min = guess_min
        self.combine_orgs = combine_orgs
        self.combine_orgs_cutoff = combine_orgs_cutoff
        self.org_most_common = use_org_most_common
        self.known_affiliations = known_affiliations
        self.attempt_fix_parse = attempt_fix_parser_errors
        if self.combine_orgs:
            self.logger.warning("combine_orgs is not yet implemented, it will have no effect")
        if self.guess_email_and_aff:
            self.logger.warning("guess_email_and_aff is not yet implemented, it will have no effect")
        if self.attempt_fix_parse:
            self.logger.warning("attempt_fix_parser_errors is not yet implemented, it will have no effect")

    def __call__(self, xml_path, debug_cutoff=None, debug_part=None):
        """
        Run the parser
        :param xml_path: Path to the parsed pdfs
        :param debug_cutoff: Only for debugging purposes
        :param debug_part: Part to debug
        :return: dict of parsed_papers
        """

        parser = PDFParser(aliases=self.aliases, id_to_name=self.id_to_name,
                           same_names=self.same_names, sim_cutoff=self.similarity_cutoff)

        if xml_path[-1] != '/':
            xml_path = xml_path + '/'
        parsed_pdfs = [f for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and ".xml" in f]
        if debug_cutoff:
            parsed_pdfs = parsed_pdfs[:debug_cutoff]

        # Check if files are already parsed
        to_use = []
        to_check = []
        if len(self.parsed) != 0:
            printLogToConsole(self.console_log_level, "Removing already parsed papers", logging.INFO)
            self.logger.info("Removing already parsed papers")
            for p in parsed_pdfs:
                try:
                    if p.split(".")[0] in self.parsed:
                        self.logger.debug("{} was found in parsed".format(p))
                        to_check.append(p)
                    else:
                        to_use.append(p)
                except Exception as e:
                    self.logger.debug("p.split(\".\")[0] failed for {}".format(p))
                    self.logger.exception(e)
        else:
            to_use = parsed_pdfs
        self.logger.debug("to_check={}".format(to_check))
        self.logger.debug("to_use={}".format(len(to_use)))
        parsed_pdfs = to_use

        if debug_part is not None and debug_part == "remove_parsed":
            return to_use, to_check, parsed_pdfs

        # Set up variables for later stats/errors stuff
        errors = []
        errors_pre_parse = 0
        papers_before_parse = 0
        warnings = []
        manual_fixes_needed = {}
        total_correct_manual = 0
        total_fixed = 0

        printLogToConsole(self.console_log_level, "Preparing PDF args", logging.INFO)
        self.logger.log(logging.INFO, "Preparing PDF args")
        raw_results = []
        args = []

        args_pbar = tqdm(total=len(parsed_pdfs), file=sys.stdout, dynamic_ncols=True)
        for f in parsed_pdfs:
            try:
                pid = f.split(".")[0]
            except IndexError as e:
                args_pbar.update()
                errors.append("ERROR:{}: failed to get the pid".format(f))
                self.logger.warning("{}'s failed to get pid".format(f))
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
                        errors.append("{}'s xml could not be read".format(pid))
                        self.logger.warning("{}'s xml could not be read".format(pid))
                        continue
            except:
                args_pbar.update()
                errors_pre_parse += 1
                errors.append("{}'s xml could not be opened".format(pid))
                self.logger.warning("{}'s xml could not be opened ".format(pid))
                continue

            if pid in self.manual_fixes:
                man_fixes = self.manual_fixes[pid]
            else:
                man_fixes = {}
            papers_before_parse += 1
            args.append([current_paper, root, man_fixes])
            args_pbar.update()
        args_pbar.close()
        if debug_part is not None and debug_part == "open_xml":
            return args

        # Determine if we should use more than one core
        t_parse_start = time.time()
        if self.cores == 1 or len(args) < self.parse_parallel_cutoff:
            if debug_part is not None and debug_part == "cores":
                return 1, len(args)
            printLogToConsole(self.console_log_level, "Parsing papers on a single core", logging.INFO)
            self.logger.info("Parsing papers on a single core")
            if len(args) < self.parse_parallel_cutoff:
                self.logger.debug("args was less than self.parse_parallel_cutoff ({} < {})".format(len(args),
                                                                                                   self.parse_parallel_cutoff))

            parse_pbar = tqdm(total=len(args), file=sys.stdout)
            for i in args:
                raw_results.append(parser(i))
                parse_pbar.update()
            parse_pbar.close()

        else:

            self.logger.debug("Parsing in parallel with {} cores".format(self.cores))
            batches = chunks(args, self.batch_size)
            batch_count, rem = divmod(len(args), self.batch_size)
            if rem != 0:
                batch_count += 1
            if debug_part is not None and debug_part == "cores":
                return batch_count
            printLogToConsole(self.console_log_level, "Parsing {} batches".format(batch_count), logging.INFO)
            self.logger.info("Parsing {} batches".format(batch_count))
            with mp.Pool(self.cores) as Pool:
                imap_results = list(
                    tqdm(Pool.imap_unordered(parser.parseBatch, batches), total=batch_count, file=sys.stdout))
            for i in imap_results:
                raw_results.extend(i)
            if not raw_results:
                self.logger.error("Nothing in raw_results")
                raise Exception("raw_results is empty")
        t_parse_end = time.time()
        try:
            papers_per_second = len(args) / (t_parse_end - t_parse_start)
        except ZeroDivisionError:
            papers_per_second = len(args)
        self.logger.debug("{:.2f} papers/second".format(papers_per_second))

        printLogToConsole(self.console_log_level, "Handling raw results", logging.INFO)
        self.logger.info("Handling raw results")
        raw_pbar = tqdm(total=len(raw_results), file=sys.stdout)
        for rtr, status, error_msgs in raw_results:
            if status == 0:
                paper_out, man_fixes, fixed, corrected = rtr
                self.parsed[paper_out.pid] = paper_out
                if man_fixes:
                    manual_fixes_needed[paper_out.pid] = man_fixes
                total_fixed += fixed
                total_correct_manual += corrected
                warnings.extend(error_msgs)
                for i in error_msgs:
                    self.logger.warning(i)
            else:
                warnings.extend(error_msgs)
                for e in error_msgs:
                    if self.print_errors:
                        self.logger.error(e)
                    else:
                        self.logger.warning(e)
            raw_pbar.update()
        raw_pbar.close()
        if debug_part is not None and debug_part == "parse":
            return self.parsed

        self._getOrgsAndDep()

        # Display results
        results = [
            ["Papers parsed/second", papers_per_second],
            ["Total Papers Parsed", len(self.parsed)],
            ["Failures", papers_before_parse - len(self.parsed)],
            ["Manual Fixes Needed", len(manual_fixes_needed)],
            ["Total Fixes", total_fixed],
            ["Corrected with manual fixes", total_correct_manual],
            ["Warnings", len(warnings)],
            ["Unique organization names", len(self.org_names)],
            ["Unique department names", len(self.department_names)],
            ["Unique Authors", len(self.author_papers)]
        ]
        self.logger.debug("Results from parsing:")
        for msg, value in results:
            self.logger.debug("\t{}: {}".format(msg, value))
        printStats("PDF Parsing Stats", results)

        printLogToConsole(self.console_log_level, "Generating Tokenized for titles", logging.INFO)
        self.logger.log(logging.INFO, "Generating Tokenized for titles")
        if self.cores == 1:
            with tqdm(file=sys.stdout, total=len(self.parsed)) as pbar:
                for k in self.parsed.keys():
                    self.parsed[k].createPOS()
                    pbar.update()
                pbar.close()
        else:
            with mp.Pool(self.cores) as Pool:
                pool_results = list(tqdm(Pool.imap_unordered(self._getTokenized, [x for k, x in self.parsed.items()]),
                                         total=len(self.parsed), file=sys.stdout))
            for k, t, c, s in pool_results:
                self.parsed[k].loadTokenized(t, c, s)
        for k in manual_fixes_needed.keys():
            self.incomplete_papers.append(k)
        if self.save_data:
            self._saveData(manual_fixes_needed)
        return self.parsed

    def _saveData(self, manual_fixes_needed):
        """
        Saves files if save_data is True
        :param manual_fixes_needed: manual fixes needed after parsing
        :return: None
        """

        json_path = self.save_dir
        csv_path = self.save_dir
        txt_path = self.save_dir

        if self.ext_directory:
            json_path = json_path + "/json"
            csv_path = csv_path + "/csv"
            txt_path = txt_path + "/txt"
            if not os.path.exists(json_path):
                os.mkdir(json_path)
            if not os.path.exists(csv_path):
                os.mkdir(csv_path)
            if not os.path.exists(txt_path):
                os.mkdir(txt_path)

        printLogToConsole(self.console_log_level, "Writing parsed papers", logging.INFO)
        self.logger.log(logging.INFO, "Writing parsed papers")
        write_papers_pbar = tqdm(total=len(self.parsed), file=sys.stdout)
        papers_print = {}
        for x in self.parsed.keys():
            papers_print[x] = self.parsed[x].asDict()
            write_papers_pbar.update()
        write_papers_pbar.close()
        with open(json_path + "/parsed_papers.json", "w") as f:
            ujson.dump(papers_print, f)

        with open(json_path + "/organizations.json", "w") as f:
            json.dump(self.organizations, f, indent=4, sort_keys=True)
        with open(json_path + "/effective_org_info.json", "w") as f:
            json.dump(self.effective_org_info, f, indent=4)
        with open(json_path + "/author_papers.json", "w") as f:
            json.dump(self.author_papers, f, indent=4)

        printLogToConsole(self.console_log_level, "Writing manual fixes needed", logging.INFO)
        self.logger.log(logging.INFO, "Writing manual fixes needed")
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

        with open(txt_path + "/incomplete_papers.txt", "w") as f:
            for p in self.incomplete_papers:
                f.write(p + "\n")
        with open(txt_path + "/org_corpus.txt", "w") as f:
            for p in self.org_names:
                f.write(p + "\n")
        with open(txt_path + "/department_corpus.txt", "w") as f:
            for p in self.department_names:
                f.write(p + "\n")

        if json_path != csv_path:
            printLogToConsole(self.console_log_level, "Wrote json files to {}".format(json_path), logging.INFO)
            self.logger.log(logging.INFO, "Wrote json files to {}".format(json_path))
            printLogToConsole(self.console_log_level, "Wrote csv files to {}".format(csv_path), logging.INFO)
            self.logger.log(logging.INFO, "Wrote csv files to {}".format(csv_path))
            printLogToConsole(self.console_log_level, "Wrote txt files to {}".format(txt_path), logging.INFO)
            self.logger.log(logging.INFO, "Wrote txt files to {}".format(txt_path))
        else:
            printLogToConsole(self.console_log_level, "Wrote files to {}".format(self.save_dir), logging.INFO)
            self.logger.log(logging.INFO, "Wrote files to {}".format(self.save_dir))

    def _getOrgsAndDep(self):
        printLogToConsole(self.console_log_level, "Getting organizations and departments", logging.INFO)
        self.logger.info("Getting organizations and departments")
        org_corpus = []
        people_orgs = defaultdict(list)
        tokenizer = whitespace_tokenizer.WhitespaceTokenizer()
        tmp_organizations_info = {}
        org_first_letter = defaultdict(Counter)
        address_keys = ["postCode", "region", "settlement", "country"]
        org_pbar = tqdm(total=len(self.parsed), file=sys.stdout)
        for p, paper in self.parsed.items():
            for a, aff_email in paper.affiliations.items():
                self.author_papers[a].append(p)
                aff = aff_email["affiliation"]
                org_id = aff["id"]
                if aff["type"] and org_id:
                    if org_id not in tmp_organizations_info:
                        if org_id in self.organizations:
                            tmp_organizations_info[org_id] = self.organizations[org_id]
                        else:
                            tmp_organizations_info[org_id] = {
                                "name": Counter(),
                                "type": Counter(),
                                "postCode": Counter(),
                                "region": Counter(),
                                "settlement": Counter(),
                                "country": Counter(),
                                "count": 0
                            }
                    org_name = cleanName(aff["info"][aff["type"][0]][0])
                    self.org_names.append(org_name)
                    tmp_organizations_info[org_id]["name"][org_name] += 1
                    tmp_organizations_info[org_id]["type"][aff["type"][0]] += 1
                    tmp_organizations_info[org_id]["count"] += 1
                    org_corpus.append(tokenizer.tokenize(org_name))
                    org_first_letter[org_id[0]][org_id] += 1

                    people_orgs[org_id].append((p, a))
                    address = aff["address"]
                    for k in address.keys():
                        if k not in address_keys:
                            continue
                        try:
                            try:
                                tmp_address = remove_punct.sub("", address[k].split(","))
                            except Exception as e:
                                tmp_address = [None]
                            tmp_organizations_info[org_id][k][address[k]] += 1
                        except Exception as e:
                            self.logger.error("{} was raised".format(e))
                            self.logger.error("org_id = {}".format(org_id))
                            self.logger.error("address = {}".format(address))
                            self.logger.error("tmp_org_info[{}] = {}".format(org_id, tmp_organizations_info[org_id]))
                            self.logger.error("k = {}".format(k))
                            raise e
                if aff["info"]:
                    if aff["info"]["department"]:
                        for i in aff["info"]["department"]:
                            self.department_names.append(cleanName(i))
            org_pbar.update()
        org_pbar.close()
        printLogToConsole(self.console_log_level, "Combining information in each organization", logging.INFO)
        self.logger.info("Combining information in each organization")
        if self.cores == 1:
            org_pbar = tqdm(total=len(tmp_organizations_info), file=sys.stdout)
            for k in tmp_organizations_info.keys():
                self.organizations[k] = self._combineOrgInfo([k, tmp_organizations_info[k]])
                org_pbar.update()
            org_pbar.close()
        else:
            with mp.Pool(self.cores) as Pool:
                org_args = [[k, v] for k, v in tmp_organizations_info.items()]
                res = list(
                    tqdm(Pool.imap_unordered(self._combineOrgInfo, org_args), total=len(org_args), file=sys.stdout))
            for k, r in res:
                self.organizations[k] = r
        # TODO: Implement way to combine orgs
        self.org_names = list(set(self.org_names))
        self.department_names = list(set(self.department_names))
        self.logger.debug("{} unique departments".format(len(self.department_names)))
        self.logger.debug("{} unique organizations".format(len(self.org_names)))
        if self.org_most_common:
            printLogToConsole(self.console_log_level, "Using most common values for organizations", logging.INFO)
            self.logger.info("Using most common values for organizations")
            fix_pbar = tqdm(total=len(tmp_organizations_info), file=sys.stdout)
            authors_affected = []
            for org, info in tmp_organizations_info.items():
                if not org:
                    fix_pbar.update()
                    continue
                org_info = {
                    "type": info["type"].most_common(1)[0][0],
                    "name": info["name"].most_common(1)[0][0]
                }
                for k in address_keys:
                    if k in info:
                        try:
                            org_info[k] = info[k].most_common(1)[0][0]
                        except IndexError:
                            org_info[k] = None
                    else:
                        org_info[k] = None
                self.effective_org_info[org] = org_info
                for p, a in people_orgs[org]:
                    old_aff = deepcopy(self.parsed[p].affiliations[a]["affiliation"])
                    try:
                        old_type = old_aff["type"][0]
                    except IndexError:
                        old_type = None

                    old_aff["type"] = [org_info["type"]]
                    if old_type:
                        del old_aff["info"][old_type]
                    old_aff["info"][org_info["type"]] = [org_info["name"]]
                    new_address = {}
                    for k in address_keys:
                        new_address[k] = org_info[k]
                    old_aff["address"] = new_address
                    self.parsed[p].affiliations[a]["affiliation"] = deepcopy(old_aff)
                    authors_affected.append([p, a])
                fix_pbar.update()
            fix_pbar.close()
            self.logger.debug("{} Authors affected".format(len(authors_affected)))
            self.logger.debug("{} First 10 affected".format(authors_affected[:10]))

        else:
            self.effective_org_info = tmp_organizations_info

    def _validatePaper(self, paper_dict):
        # TODO: Implement this
        return True

    @staticmethod
    def _getTokenized(p):
        tokenized = p.tokenize()
        return [p.pid, *tokenized]

    @staticmethod
    def _combineOrgInfo(args):
        states = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
        }
        org_id, info = args
        out = {}
        for k, v in info.items():
            if k != "count" and k != "type" and k != "name":
                orig_count = sorted(v.items(), key=lambda x: x[1], reverse=True)
                unique_elements = {}
                new_counter = Counter()
                for i, c in orig_count:
                    if i is not None:

                        all_tokens = split_address.split(i)
                        for token in all_tokens:
                            if token is None or not token:
                                continue

                            token = token.strip()
                            if token in states:
                                t = states[token]
                            else:
                                t = token
                            if t in unique_elements:
                                new_counter[t] += c
                            elif t.lower() in unique_elements:
                                new_counter[unique_elements[t.lower()]] += c
                            else:
                                all_elements = "".join([x[0] for x in new_counter.items()])
                                try:
                                    close_match = fuzzysearch.find_near_matches(t, all_elements, max_l_dist=1)
                                except Exception as e:
                                    raise e
                                if len(close_match) == 0:
                                    close_match = fuzzysearch.find_near_matches(t.lower(), all_elements, max_l_dist=1)
                                    if len(close_match) == 0:
                                        close_match = []
                                        unique_elements[t.lower()] = t
                                        new_counter[t] += c
                                matches_already_used = set()
                                for m in close_match:
                                    match = all_elements[m.start:m.end]
                                    if match in matches_already_used:
                                        continue
                                    new_counter[match] += c
                                    matches_already_used.add(match)

                    else:
                        new_counter[i] += c
                out[k] = new_counter
            else:
                out[k] = v
        return org_id, out
