import json
from src.acl_parser import ACLParser
from src.pdf_parser import PDFParserWrapper
from src.config_handler import ConfigHandler
from src.utility_functions import createCLIGroup, parseCLIArgs, loadData,createCLIShared
import os
import gc
import argparse

arguments = argparse.ArgumentParser(
    description="Parse ACL files and parsed PDF xml files. You can specify these in config.json instead of using "
                "command line arguments",
    formatter_class=argparse.MetavarTypeHelpFormatter)
createCLIShared(arguments)
createCLIGroup(arguments, "PDFParser",
               "Arguments for the PDFParser, check the documentation of pdf_parser.py to see default values",
               PDFParserWrapper.parameters)
createCLIGroup(arguments, "ACLParser",
               "Arguments for the ACLParser, check the documentation of acl_parser.py to see default values",
               ACLParser.parameters)

if __name__ == '__main__':
    args = arguments.parse_args()
    with open(os.getcwd() + "/logs/create_data.log", 'w'):
        pass
    log_path = os.getcwd() + "/logs/create_data.log"
    print("INFO: Starting Create Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "create_data", raise_error_unknown=True)
    config = parseCLIArgs(args, config)
    acl_parser = ACLParser(**config["ACLParser"])
    acl_parser(config["xml_path"], config["name_variants_path"])

    data = loadData(["aliases","acl_papers","id_to_name","same_names"],config.logger,config,override_keys={"acl_papers":"papers"})
    parser = PDFParserWrapper(**data,
                              **config["PDFParser"])
    parser(config["parsed_pdf_path"])
    gc.collect()
