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
        "cores",
        "disable_pbars",
        "pbar_ascii"
    ]
    pdf_parser_keys = [
        "load_parsed",
        "allow_load_parsed_errors",
        "assign_similarity_cutoff",
        "print_errors",
        "parse_parallel_cutoff",
        "parse_batch_size",
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
    target_creator_keys = [
        "treat_id_different_people",
        "raise_error",
        "skip_error_papers",
        "one_target_per_paper"
    ]
    input_handler_keys= [
        "target_path"
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
    dont_overwrite = [
        "PDFParserXpaths",
        "ACLParserXpaths",
        "save_path",
        "ext_directory",
        "log_path",
        "log_format",
        "console_log_level",
        "file_log_level",
        "DEBUG_MODE",
        "disable_pbar",
    ]
    to_readable = {
        "xpath_config": "ACLParserXpaths"
    }

    def __init__(self, config_dict, log_file, file_log_level=logging.DEBUG, console_log_level=logging.WARNING,
                 log_format=None, raise_error_unknown=False):
        self.config_dict = {}
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
            self.config_dict["log_format"] = log_format
        if "log path" not in config_dict:
            log_path = os.getcwd() + "/logs/{}.log".format(log_file)
            self.config_dict["log path"] = log_path
        else:
            log_path = config_dict["log path"]
            if "\\" in log_path:
                print("ERROR: log path={}".format(log_path))
                raise ValueError("\\ in the log path, currently file paths with '\\' are not supported")
            if log_path[0] != "/":
                log_path = "/" + log_path
            if log_path[-1] != "/":
                log_path = log_path + "/"
            log_path = os.getcwd() + log_path + "{}.log".format(log_file)
            del config_dict["log path"]
            self.config_dict["log path"] = log_path

        self.raise_error_unknown = raise_error_unknown
        self.logger = createLogger("config_handler", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.logger.debug("Parsing config.json")
        self.config_dict = {**self.config_dict, **config_dict}
        self.configs = {
            "shared": {},
            "pdf_parser": {},
            "acl_parser": {},
            "create_training": {},
            "vote_classifier": {},
            "author_disambiguation": {},
            "target_creator": {},
            "input_handler": {},
            "paths": {},
        }
        self.configs["shared"]["log_path"] = self.config_dict["log path"]
        for k, v in config_dict.items():
            if k == "log path":
                continue
            tmp_k = k.replace(" ", "_")
            if tmp_k in self.excluded_keys:
                self.logger.debug("{} is in excluded, skipping it".format(tmp_k))
            else:
                self.addArgument(tmp_k, v)
        if "save_path" not in self.configs["shared"]:
            raise KeyError("save_path is not in shared config")
        self._createExtraPaths()

    def addArgument(self, key, value, override_config=False):

        configs = []
        key = key.replace(" ", "_")
        if key in self.dont_overwrite and override_config:
            self.logger.warning(
                "{} is in excluded and you are trying to override it, original value will be used".format(key))
        if key in self.shared_keys:
            if "path" in key:
                if value[-1] != "/":
                    value = value + "/"
                if os.getcwd() not in value:
                    if value[0] != "/":
                        value = "/" + value
                    value = os.getcwd() + value
            configs.append("shared")

        if key in self.pdf_parser_keys:
            configs.append("pdf_parser")

        if key in self.acl_parser_keys:
            if key == "ACLParserXpaths":
                key = "xpath_config"
            configs.append("acl_parser")

        if key in self.create_training_data_keys:
            configs.append("create_training")

        if key in self.vote_classifier_keys:
            configs.append("vote_classifier")

        if key in self.author_disambiguation_keys:
            configs.append("author_disambiguation")

        if key in self.target_creator_keys:
            configs.append("target_creator")
        if key in self.input_handler_keys:
            configs.append("input_handler")

        if key in self.path_keys:
            if "\\" in value:
                self.logger.error("{}={}".format(key, value))
                raise ValueError("\\ in the {}, currently file paths with '\\' are not supported".format(key))
            if os.getcwd() not in value:
                if value[0] != "/":
                    value = "/" + value
                value = os.getcwd() + value
            if value[-1] != "/":
                value = value + "/"
            configs.append("paths")

        if len(configs) == 0:
            if self.raise_error_unknown:
                raise ValueError("{} is not a valid argument".format(key))
            else:
                self.logger.warning("{} is not a valid argument, value will be ignored".format(key))
        else:
            for config in configs:
                if key in self.configs[config] and not override_config:
                    self.logger.debug(
                        "{} was not added to {} because it already was there and override_config=False".format(key,
                                                                                                               config))
                    continue
                self.logger.debug("{} added to config {} with value {}".format(key, config, value))
                self.configs[config][key] = value
            if key in self.dont_overwrite:
                return
            if (key in self.config_dict and override_config) or key not in self.config_dict:
                if key in self.to_readable:
                    key = self.to_readable[key]
                self.logger.debug("Added {} to config dict".format(key.replace("_", " ")))
                self.config_dict[key.replace("_", " ")] = value

    def __getitem__(self, item):
        if item == "PDFParser":
            return {**self.configs["shared"], **self.configs["pdf_parser"]}
        elif item == "ACLParser":
            return {**self.configs["shared"], **self.configs["acl_parser"]}
        elif item == "CreateTrainingData":
            return {**self.configs["shared"], **self.configs["create_training"]}
        elif item == "VoteClassifier":
            return {**self.configs["shared"], **self.configs["vote_classifier"]}
        elif item == "AuthorDisambiguation":
            return {**self.configs["shared"], **self.configs["author_disambiguation"]}
        elif item == "TargetCreator":
            return {**self.configs["shared"], **self.configs["target_creator"]}
        elif item == "InputHandler":
            return {**self.configs["shared"], **self.configs["input_handler"]}
        elif item in self.configs["paths"]:
            return self.configs["paths"][item]
        else:
            raise KeyError("{} is not a valid key to use with []".format(item))

    def _createExtraPaths(self):
        files = ["id_to_name.json", "parsed_papers.json", "aliases.json", "same_names.txt", "acl_papers.json",
                 "incomplete_papers.txt", "department_corpus.txt", "org_corpus.txt", "conflicts.json",
                 "organizations.json", "effective_org_info.json", "author_papers.json", "similar_names.json",
                 "known_affiliations.json","test_special_keys.txt","conflict_author_parsed.txt","tagged_pairs.pickle"]

        for f in files:
            file_name, extension = f.split(".")
            if "ext_directory" in self.configs["shared"] and self.configs["shared"]["ext_directory"]:
                self.configs["paths"][file_name] = self.configs["shared"]["save_path"] + "{}/{}".format(extension, f)
            else:
                self.configs["paths"][file_name] = self.configs["shared"]["save_path"] + "{}".format(f)

    def save(self):
        self.logger.debug("Saving config for future use")
        to_use = {}
        for k, v in self.config_dict.items():
            if k in self.to_readable:
                to_use[self.to_readable[k]] = v
            else:
                to_use[k] = v
        with open("config.json", "w") as f:
            json.dump(to_use, f, indent=4, sort_keys=True)

