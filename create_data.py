import json
from src.acl_parser import ACLParser
from src.pdf_parser import PDFParserWrapper
from src.config_handler import ConfigHandler
import os
import gc
import argparse
arguments = argparse.ArgumentParser(description="Parse ACL files and parsed PDF xml files")
shared_group = arguments.add_argument_group("Universal","Universal Arguments")
shared_group.add_argument("")

if __name__ == '__main__':
    with open(os.getcwd() + "/logs/create_data.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/create_data.log"
    print("INFO: Starting Create Data")
    gc.collect()
    config_raw= json.load(open("config.json"))
    config = ConfigHandler(config_raw,"create_data",raise_error_unknown=True)
    acl_parser = ACLParser(**config["ACLParser"])
    acl_parser(config["xml_path"],config["name_variants_path"])

    aliases = json.load(open(config["aliases"]))
    papers = json.load(open(config["acl_papers"]))
    id_to_name = json.load(open(config["id_to_name"]))
    same_names = [x.strip() for x in open(config["same_names"]).readlines()]
    parser = PDFParserWrapper(papers=papers, aliases=aliases, id_to_name=id_to_name, same_names=same_names,**config["PDFParser"])
    parser(config["parsed_pdf_path"])
    gc.collect()