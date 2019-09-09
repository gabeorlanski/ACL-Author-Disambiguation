from src.utility_functions import createCLIGroup, createCLIShared, createLogger, loadData
from src.target_creator import TargetCreator
from src.author_disambiguation import AuthorDisambiguation
from src.input_handler import InputHandler
from src.paper import Paper
from src.config_handler import ConfigHandler
import json
import logging
import numpy as np
import os
from nltk import PorterStemmer
import gc
import argparse
from tqdm import tqdm
import random
import sys

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
    with open(os.getcwd() + "/logs/evaluate_disambiguation.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/evaluate_disambiguation.log"
    print("INFO: Starting Create Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "evaluate_disambiguation", raise_error_unknown=True)
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

    target_creator = TargetCreator(parsed, id_to_name, author_papers, **config["TargetCreator"])
    tests = [
        "yang-liu-ict",
        "yang-liu-icsi",
        "james-allen",
        'alessandro-moschitti',
        'ahmed-abdelali',
        'chen-li'
    ]
    test_targets = []
    targets = []
    for k in author_papers.keys():
        usable_papers = [x for x in author_papers[k] if x in parsed]
        if len(usable_papers) < 3:
            continue
        targets.extend(target_creator.createTarget(k, random.sample(usable_papers,1)))
        test_targets.append(k)
        # rtr = input_handler.handleInput(k, test_papers[k])
    target_papers, target_authors, target_ids = target_creator.fillData()
    compare_authors_args = {
        "company_corpus": org_corpus,
        "department_corpus": department_corpus,
        "threshold": .4,
        "str_algorithm":["jaro","similarity"]
    }
    disambiguation = AuthorDisambiguation(papers=target_papers, author_papers=target_authors, compare_args=compare_authors_args, id_to_name=target_ids,
                                          **config["AuthorDisambiguation"])

    results = disambiguation(targets)

    correct = 0
    false_positives =0
    no_same = 0
    no_different = 0
    different_auth_count = []
    none_found = []
    wrong = []
    print("INFO: Evaluating results")
    pbar = tqdm(total=len(results),file=sys.stdout)
    for k,info in results.items():
        actual_k = k[:-1]
        if actual_k not in test_targets:
            raise ValueError("{} not in test_targets".format(actual_k))
        if len(info["different"]) == 0:
            no_different += 1
            pbar.update()
        else:
            different_auth_count.append(len(info["different"]))
        if info["same"] is None:
            no_same +=1
            none_found.append(actual_k)
        else:
            if actual_k != info["same"]:
                false_positives+=1
                wrong.append(actual_k)
            else:
                correct +=1


        pbar.update()
    pbar.close()
    print(none_found)
    print(wrong)
    print("INFO: {} targets were unable to find a same author".format(no_same))
    print("INFO: {} had more than 1 different author".format(len(different_auth_count)))
    precision = correct/(correct+false_positives)
    print("INFO: Precision = {:.2f}".format(precision*100))
    recall = correct/(correct+no_same)
    print("INFO: Recall = {:.2f}".format(recall*100))
    try:
        f1 = 2/(1/precision+1/recall)
    except ZeroDivisionError:
        f1 = 0
    print("INFO: F1 Score = {:.2f}".format(f1*100))
    print("INFO: People with no different authors = {}".format(no_different))
    avg_diff = np.mean(different_auth_count)
    print("INFO: Average different authors = {:.2f}".format(avg_diff))
    with open("test_results.txt","a") as f:
        f.write("Wrong author = {}\n".format(wrong))
        f.write("No author = {}\n".format(none_found))
        f.write("Precision = {:.2f}\n".format(precision*100))
        f.write("Recall = {:.2f}\n".format(recall*100))
        f.write("F1 Score = {:.2f}\n".format(f1*100))


