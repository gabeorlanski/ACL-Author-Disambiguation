from src.utility_functions import createCLIGroup, createCLIShared, createLogger, loadData
from src.target_creator import TargetCreator
from src.author_disambiguation import AuthorDisambiguation
from src.input_handler import InputHandler
from src.paper import Paper
from src.config_handler import ConfigHandler
import json
import logging
import os
from nltk import PorterStemmer
import gc
import argparse

stemmer = PorterStemmer()
arguments = argparse.ArgumentParser(
    description="Parse Disambiguate targets. You can specify these in config.json instead of using command line arguments",
    formatter_class=argparse.MetavarTypeHelpFormatter)
createCLIShared(arguments)
createCLIGroup(arguments, "TargetCreator", "Arguments for how to create targets", TargetCreator.parameters)
createCLIGroup(arguments, "AuthorDisambiguation", "Arguments for how to disambiguate authors, check author_disambiguation.py for default values",
               AuthorDisambiguation.parameters)

if __name__ == '__main__':
    args = arguments.parse_args()
    with open(os.getcwd() + "/logs/disambiguate.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/disambiguate.log"
    print("INFO: Starting Create Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "disambiguate", raise_error_unknown=True)
    data = loadData(
        ["department_corpus", "incomplete_papers", "org_corpus", "conflicts", "parsed_papers", "same_names", "test_special_keys", "author_papers",
         "id_to_name"], config.logger, config)
    author_papers = data["author_papers"]
    id_to_name = data["id_to_name"]
    same_names = data["same_names"]
    parsed = data["parsed_papers"]
    parsed = {x: Paper(**info) for x, info in parsed.items()}
    org_corpus = data["org_corpus"]
    department_corpus = data["department_corpus"]
    incomplete = data["incomplete_papers"]
    special_keys = data["test_special_keys"]
    input_handler = InputHandler(parsed, author_papers, id_to_name, **config["InputHandler"])
    input_handler.handleUserInput()
    target_creator = TargetCreator(parsed, id_to_name, author_papers)
    targets = []
    for k in input_handler.targets:
        for p in author_papers[k]:
            if k not in parsed[p].affiliations:
                continue
            targets.append(target_creator.createTarget(k, [p]))
        # rtr = input_handler.handleInput(k, test_papers[k])
    compare_authors_args = {
        "company_corpus": org_corpus,
        "department_corpus": department_corpus,
        "threshold": .4
    }
    disambiguation = AuthorDisambiguation(target_creator.papers, target_creator.author_papers, compare_authors_args, target_creator.id_to_name,
                                          **config["AuthorDisambiguation"])

    results = disambiguation(targets)
