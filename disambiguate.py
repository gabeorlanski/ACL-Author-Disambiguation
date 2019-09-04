from src.utility_functions import createCLIGroup, createCLIShared, createLogger, loadData
from src.target_creator import TargetCreator
from src.author_disambiguation import AuthorDisambiguation
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
               TargetCreator.parameters)


if __name__ == '__main__':
    args = arguments.parse_args()
    with open(os.getcwd() + "/logs/disambiguate.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/create_data.log"
    print("INFO: Starting Create Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "disambiguate", raise_error_unknown=True)
