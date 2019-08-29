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

if __name__ == '__main__':
    with open(os.getcwd() + "/logs/create_data.log", 'w'):
        pass

    print("INFO: Starting Create Data")
    gc.collect()
    config = json.load(open("config.json"))
    data_path = os.getcwd() + "/data"

    save_path = os.getcwd() + config["save data path"]
    acl_parser = ACLParser(config["ACLParserXpaths"],save_data=True,save_path=save_path,ext_directory=True)
    acl_parser(os.getcwd()+config["xml path"],os.getcwd()+config["name variants path"])

    aliases = json.load(open(data_path + "/json/aliases.json"))
    papers = json.load(open(data_path + "/json/acl_papers.json"))
    id_to_name = json.load(open(data_path + "/json/id_to_name.json"))
    same_names = [x.strip() for x in open(data_path + "/txt/same_names.txt").readlines()]
    parser = PDFParserWrapper(config["parsed pdf path"],save_data=True,save_dir=save_path,ext_directory=True,cores=1,batch_size=500)
    parser.loadData(papers, aliases, id_to_name, same_names,{})
    parser(os.getcwd() + "/data/pdf_xml")
    gc.collect()