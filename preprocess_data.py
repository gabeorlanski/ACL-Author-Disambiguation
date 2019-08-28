import json
from src.acl_parser import ACLParser
from src.pdf_parser import PDFParserWrapper
from src.create_training_data import CreateTrainingData
from src.paper import Paper
from src.utility_functions import createLogger
import os
import gc
from nltk.stem import PorterStemmer
import pickle
stemmer = PorterStemmer()

if __name__ == "__main__":
    with open(os.getcwd() + "/logs/preprocess_data.log", 'w'):
        pass

    print("INFO: Starting Data processing")
    gc.collect()
    config = json.load(open("config.json"))
    data_path = os.getcwd() + "/data"

    save_path = os.getcwd() + config["save data path"]
    # acl_parser = ACLParser(config["ACLParserXpaths"],save_data=True,save_path=save_path,ext_directory=True)
    # acl_parser(os.getcwd()+config["xml path"],os.getcwd()+config["name variants path"])

    aliases = json.load(open(data_path + "/json/aliases.json"))
    papers = json.load(open(data_path + "/json/acl_papers.json"))
    id_to_name = json.load(open(data_path + "/json/id_to_name.json"))
    same_names = [x.strip() for x in open(data_path + "/txt/same_names.txt").readlines()]
    # parser = PDFParserWrapper(config["parsed pdf path"],save_data=True,save_dir=save_path,ext_directory=True,cores=1,batch_size=500)
    # parser.loadData(papers, aliases, id_to_name, same_names,{})
    # parser(os.getcwd() + "/data/pdf_xml")
    # gc.collect()
    parsed = json.load(open(save_path + "/json/parsed_papers.json"))
    parsed = {x: Paper(**info) for x, info in parsed.items()}
    org_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                  open(save_path + "/txt/org_corpus.txt").readlines()]
    department_corpus = [[stemmer.stem(w) for w in x.strip().split()] for x in
                         open(save_path + "/txt/department_corpus.txt").readlines()]
    incomplete = [x.strip() for x in open(save_path + "/txt/incomplete_papers.txt").readlines()]

    compare_authors_args = {
        "company_corpus": org_corpus,
        "department_corpus": department_corpus,
        "threshold": .4
    }
    excluded_dict = json.load(open(data_path+"/json/conflicts.json"))
    excluded = []
    for k, c in excluded_dict.items():
        for _id,n in excluded:
            excluded.append(_id)
    special_keys = [x.strip() for x in open("test_special_keys.txt").readlines() if x != "\n"]
    # saved_pairs = pickle.load(open("saved_pairs.pickle","rb"))
    # pairs_to_use = []
    # for k,t in saved_pairs:
    #     k_split = k.split()
    #     a = " ".join(k_split[:2])
    #     b = " ".join(k_split[2:])
    #     pairs_to_use.append([t,[k,a,b]])
    pair_creator = CreateTrainingData(parsed, incomplete, special_keys, name_similarity_cutoff=.9,
                                      author_cutoff=1, batch_size=25000, dif_same_ratio=1.2, separate_chars=1,
                                      compare_args=compare_authors_args,save_data=True,cores=4,compare_batch_size=1000,exclude=excluded,allow_exact_special=True)
    pair_creator(get_info_all=True)
