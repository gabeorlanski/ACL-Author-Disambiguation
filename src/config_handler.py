import json
import os
import sys
import logging
from src.utility_functions import createLogger


class ConfigHandler:
    shared_keys = [
        "save_data",
        "save_path",
        "ext_directory",
        "log_path",
        "log_format",
        "console_log_level",
        "file_log_level",
        "cores"
    ]
    pdf_parser_keys = [
        "load_parsed",
        "allow_load_parsed_errors",
        "similarity_cutoff",
        "print_errors",
        "parse_parallel_cutoff",
        "batch_size",
        "guess_email_and_aff",
        "guess_min",
        "combine_orgs_cutoff",
        "use_org_most_common",
        "known_affiliations",
        "attempt_fix_parser_errors"
    ]
    acl_parser_keys = [
        "ACLParserXpaths"
    ]
    create_training_data_keys = [
        "special_keys",
        "dif_same_ratio",
        "author_cutoff",
        "name_similarity_cutoff",
        "pair_distribution",
        "separate_chars",
        "separate_words",
        "algorithm",
        "exclude",
        "rand_seed",
        "batch_size",
        "allow_exact_special",
        "min_batch_len",
        "DEBUG_MODE",
        "drop_null_authors",
        "print_compare_stats",
        "compare_args",
        "compare_batch_size",
        "remove_single_author",
        "require_exact_match",
    ]
    author_disambiguation_keys = [
        "threshold",
        "name_similarity_cutoff",
        "str_algorithm",
        "model",
        "model_name",
        "model_path",
        "skip_same_papers",
        "create_new_author",
        "compare_cutoff",
        "tie_breaker",
        "DEBUG_MODE",
        "sim_overrides",
        "allow_authors_not_in_override",
        "same_paper_diff_people",
        "use_probabilities"
    ]
    vote_classifier_keys = [
        "classifier_weights",
        "test_fraction",
        "model_save_path",
        "model_name",
        "special_cases",
        "rand_seed",
        "cutoff",
        "special_only",
        "diff_same_ratio",
        "train_all_estimators",
        "voting",
    ]
    path_keys = [
        "xml_path",
        "name_variants_path",
        "parsed_pdf_path",
        "log_path",
        "save_path"
    ]
    excluded_keys = [
        "PDFParserXpaths"
    ]
    readable_dict = {
        "directories for file type": "ext_directory",
        "save path": "save_path"
    }

    def __init__(self, config_dict, log_file, file_log_level=logging.DEBUG, console_log_level=logging.WARNING,
                 log_format=None, raise_error_unknown=False):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if "log path" not in config_dict:
            log_path = os.getcwd() + "/logs/{}.log".format(log_file)
            config_dict["log path"] = log_path
        else:
            log_path = config_dict["log path"]
            if "\\" in log_path:
                print("ERROR: log path={}".format(log_path))
                raise ValueError("\\ in the log path, currently file paths with '\\' are not supported")
            if log_path[0] !="/":
                log_path= "/"+log_path
            if log_path[-1] != "/":
                log_path = log_path + "/"
            log_path = os.getcwd()+log_path + "{}.log".format(log_file)
            config_dict["log path"] = log_path
        self.raise_error_unknown = raise_error_unknown
        self.logger = createLogger("config_handler", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.logger.debug("Parsing config.json")

        self.pdf_parser_config = {}
        self.acl_parser_config = {}
        self.create_training_config = {}
        self.vote_classifier_config = {}
        self.author_disambiguation_config = {}
        self.shared_config = {}
        self.path_config = {}
        for k, v in config_dict.items():
            if k in self.excluded_keys:
                self.logger.debug("{} is in excluded, skipping it".format(k))
            else:
                self.addArgument(k, v)
        if "save_path" not in self.shared_config:
            raise KeyError("save_path is not in shared config")
        self._createExtraPaths()

    def addArgument(self, key, value):
        if key in self.readable_dict:
            key = self.readable_dict[key]

        key = key.replace(" ", "_")
        valid_argument = False
        if key in self.shared_keys:
            if "path" in key:
                if value[-1] != "/":
                    value = value+"/"
                if os.getcwd() not in value:
                    if value[0] != "/":
                        value = "/" + value
                    value = os.getcwd()+value

            self.logger.debug("{} with value {} is in shared".format(key, value))
            self.shared_config[key] = value
            valid_argument = True

        if key in self.pdf_parser_keys:
            self.logger.debug("{} with value {} is in PDFParser".format(key, value))
            self.pdf_parser_config[key] = value
            valid_argument = True

        if key in self.acl_parser_keys:
            if key == "ACLParserXpaths":
                key = "xpath_config"
            self.logger.debug("{} with value {} is in ACLParser".format(key, value))
            self.acl_parser_config[key] = value
            valid_argument = True

        if key in self.create_training_data_keys:
            self.logger.debug("{} with value {} is in CreateTrainingData".format(key, value))
            self.create_training_config[key] = value
            valid_argument = True

        if key in self.vote_classifier_keys:
            self.logger.debug("{} with value {} is in VoteClassifier".format(key, value))
            self.vote_classifier_keys[key] = value
            valid_argument = True

        if key in self.author_disambiguation_keys:
            self.logger.debug("{} with value {} is in AuthorDisambiguation".format(key, value))
            self.author_disambiguation_config[key] = value
            valid_argument = True

        if key in self.path_keys:
            if "\\" in value:
                self.logger.error("{}={}".format(key, value))
                raise ValueError("\\ in the {}, currently file paths with '\\' are not supported".format(key))
            if os.getcwd() not in value:
                if value[0] != "/":
                    value = "/"+value
                value = os.getcwd()+value
            if value[-1] != "/":
                value = value + "/"
            self.logger.debug("Found path {} with value {}".format(key, value))
            self.path_config[key] = value
            valid_argument = True

        if not valid_argument:
            if self.raise_error_unknown:
                raise ValueError("{} is not a valid argument".format(key))
            else:
                self.logger.warning("{} is not a valid argument, value will be ignored".format(key))

    def __getitem__(self, item):
        if item == "PDFParser":
            return {**self.shared_config, **self.pdf_parser_config}
        elif item == "ACLParser":
            return {**self.shared_config, **self.acl_parser_config}
        elif item == "CreateTrainingData":
            return {**self.shared_config, **self.create_training_config}
        elif item == "VoteClassifier":
            return {**self.shared_config, **self.vote_classifier_config}
        elif item == "AuthorDisambiguation":
            return {**self.shared_config, **self.author_disambiguation_config}
        elif item in self.path_config:
            return self.path_config[item]
        else:
            raise KeyError("{} is not a valid key to use with []".format(item))

    def _createExtraPaths(self):
        files = ["id_to_name.json", "parsed_papers.json", "aliases.json", "same_names.txt", "acl_papers.json",
                 "incomplete_papers.txt", "department_corpus.txt", "org_corpus.txt", "conflicts.json",
                 "organizations.json", "effective_org_info.json", "author_papers.json", "similar_names.json",
                 "known_affiliations.json"]

        for f in files:
            file_name, extension = f.split(".")
            if "ext_directory" in self.shared_config:
                self.path_config[file_name] = self.shared_config["save_path"] + "{}/{}".format(extension, f)
            else:
                self.path_config[file_name] = self.shared_config["save_path"] + "{}".format(f)
