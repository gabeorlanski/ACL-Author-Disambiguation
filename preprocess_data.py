import json
from src.config_handler import ConfigHandler
from src.create_training_data import CreateTrainingData
from src.paper import Paper
from src.utility_functions import createCLIGroup,createCLIShared, parseCLIArgs, loadData
import os
import gc
from nltk.stem import PorterStemmer
import argparse

stemmer = PorterStemmer()
arguments = argparse.ArgumentParser(
    description="Process papers using CreateTrainingData. You can specify these in config.json instead of using "
                "command line arguments",
    formatter_class=argparse.MetavarTypeHelpFormatter)
createCLIShared(arguments)
createCLIGroup(arguments, "CreateTrainingData",
               "Arguments for the CreateTrainingData, check the documentation of CreateTrainingData to see default "
               "values",
               CreateTrainingData.parameters)

if __name__ == "__main__":
    args = arguments.parse_args()
    with open(os.getcwd() + "/logs/preprocess_data.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/preprocess_data.log"
    print("INFO: Starting Preprocess Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "preprocess_data", raise_error_unknown=True)
    config = parseCLIArgs(args, config)
    data = loadData(
        ["department_corpus", "incomplete_papers", "org_corpus", "conflicts", "parsed_papers",
         "same_names", "test_special_keys"], config.logger, config)
    same_names = data["same_names"]
    parsed = data["parsed_papers"]
    parsed = {x: Paper(**info) for x, info in parsed.items()}
    org_corpus = data["org_corpus"]
    department_corpus = data["department_corpus"]
    incomplete = data["incomplete_papers"]
    special_keys = data["test_special_keys"]
    excluded_dict = data["conflicts"]

    compare_authors_args = {
        "company_corpus": org_corpus,
        "department_corpus": department_corpus,
        "threshold": .4
    }
    excluded = []
    for k, c in excluded_dict.items():
        for _id, n in c:
            excluded.append(_id)
    config.addArgument("exclude",excluded)
    pair_creator = CreateTrainingData(parsed, incomplete, special_keys,compare_args=compare_authors_args, **config["CreateTrainingData"])
    gc.collect()
    pair_creator(get_info_all=True)
