import json
from src.acl_parser import ACLParser
from src.pdf_parser import PDFParserWrapper
from src.config_handler import ConfigHandler
from src.utility_functions import createCLIGroup, parseCLIArgs, loadData
import os
import gc
import argparse

arguments = argparse.ArgumentParser(
    description="Parse ACL files and parsed PDF xml files. You can specify these in config.json instead of using "
                "command line arguments",
    formatter_class=argparse.MetavarTypeHelpFormatter)
shared_group = arguments.add_argument_group("Universal",
                                            "Universal Arguments shared across all modules. Once you have decided on "
                                            "which arguments you want, save them so you don't need to pass them "
                                            "each time you run the program")
shared_group.add_argument("--n", dest="cores", type=int, help="Number of workers to use", default=None)
shared_group.add_argument("--out_dir", dest="save_path", type=str, default=None, help="Path to save to")
shared_group.add_argument("--ext_dir", dest="ext_dir", nargs="?", const=True, type=bool,
                          help="Create a directory for each file type",
                          default=None)
shared_group.add_argument("--d", dest="debug", type=bool, nargs="?", const=True, default=None,
                          help="Print debug messages to console. WARNING: This will mess up progress bars")
shared_group.add_argument("--log_path", dest="log_path", type=str, default=None, help="Path to log files")
shared_group.add_argument("-s", dest="save_config", nargs="?", const=True, type=bool, default=False,
                          help="Save current arguments to config.json")
shared_group.add_argument("-o", dest="overwrite_config", nargs="?", const=True, type=bool,
                          default=False,
                          help="Overwrite arguments found in config.json")
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
