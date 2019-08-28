from tqdm import tqdm
import numpy as np
from src.author_disambiguation import AuthorDisambiguation
from src.disambiguation_input_handler import DisambiguationInputHandler
import json
import os
import sys
from nltk import PorterStemmer
import random

stemmer = PorterStemmer()

if __name__ == '__main__':

    with open("test_results.txt","w") as f:
        pass
    data_path = os.getcwd() + "/data"
    org_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                  open(data_path + "/txt/org_corpus.txt").readlines()]
    department_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                         open(data_path + "/txt/department_corpus.txt").readlines()]
    incomplete = [x.strip() for x in open(data_path + "/txt/incomplete_papers.txt").readlines()]
    compare_authors_args = {
        "company_corpus": org_corpus,
        "department_corpus": department_corpus,
        "threshold": .4,
        "str_algorithm": ["jaro", "similarity"]
    }
    for test_num in range(2):
        log_path = os.getcwd() + "/logs/evaluate_disambiguation.log"
        with open(log_path, "w") as f:
            pass
        print("INFO: Test {}".format(test_num))
        author_papers = json.load(open(data_path + "/json/author_papers.json"))
        id_to_name = json.load(open(data_path + "/json/id_to_name.json"))
        papers = json.load(open(data_path + "/json/parsed_papers.json"))

        input_handler = DisambiguationInputHandler(papers, id_to_name, author_papers, log_path=log_path,
                                                   treat_id_different_people=True)

        # test_targets = [x for x in author_papers.keys() if len([p for p in author_papers[x]if p in papers])> 5 and x in id_to_name ]
        test_targets = [x.strip() for x in open("test_special_keys.txt").readlines()]
        print("INFO: {} test targets".format(len(test_targets)))
        targets = []
        for k in test_targets:
            rtr = []
            try:
                if len(author_papers[k]) > 7:
                    test_papers = random.sample(author_papers[k], 3)
                else:
                    test_papers = random.sample(author_papers[k],1)
            except:
                test_papers = random.sample(author_papers[k], 1)
            for p in test_papers:
                if k not in papers[p]["affiliations"]:
                    continue
                rtr.append(input_handler.handleInput(k, [p]))
            # rtr = input_handler.handleInput(k, test_papers[k])

            if isinstance(rtr, list):
                targets.extend(rtr)
            else:
                targets.append(rtr)
        disambiguation = AuthorDisambiguation(input_handler.papers, input_handler.author_papers, compare_authors_args,
                                              input_handler.id_to_name, log_path=log_path, threshold=.01,
                                              tie_breaker="max_percent", model_name="HardVoting")

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
                # pbar.update()
                # continue
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
        f1 = 2/(1/precision+1/recall)
        print("INFO: F1 Score = {:.2f}".format(f1*100))
        print("INFO: People with no different authors = {}".format(no_different))
        avg_diff = np.mean(different_auth_count)
        print("INFO: Average different authors = {:.2f}".format(avg_diff))
        with open("test_results.txt","a") as f:
            f.write("Test {}\n".format(test_num))
            f.write("Precision = {:.2f}\n".format(precision*100))
            f.write("Recall = {:.2f}\n".format(recall*100))
            f.write("F1 Score = {:.2f}\n".format(f1*100))


