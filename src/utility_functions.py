from collections import defaultdict
import unidecode
from html import unescape
import re
from lxml import etree
import operator as op
from functools import reduce
import logging
import numpy as np
from tqdm import tqdm
import sys
import ujson
from nltk import PorterStemmer
from copy import deepcopy
stemmer = PorterStemmer()
remove_punct_ids = re.compile("[^\w\s-]")
remove_html = re.compile("<[^>]*>")
remove_punct = re.compile("[^\w\s]")


def printStats(name, to_print, leading_char="-", indents=2, decimal_cutoff=3, line_char="=", line_width=20,
               line_adaptive=False, padding=2, default_width=6, print_func=None, printing_file=False):
    """
    Function to cleanly print stats
    :param name: Name to print at beginnning
    :param to_print: list of tuples or lists, where first element is the name of the stats, and following are the stats
    themselves
    :param leading_char: What should come before any stat line is printed
    :param indents: How many indents to print with
    :param decimal_cutoff: # of decimal places to cutoff at
    :param line_char: Char to create lines with
    :param line_width: default with for each element in the lines
    :param line_adaptive: if you don't want a predifined line width, use this
    :param padding: Amount of padding for each element
    :param default_width: the default width of each element
    :param print_func: function to print with, defaults to print
    :param printing_file: file to print stats to
    :return: None
    """
    if not print_func:
        print_func = print
    column_width = defaultdict(lambda: default_width + padding)
    for s in to_print:
        for i in range(len(s)):
            v = str(s[i])
            if isinstance(s[i], float):
                v = "{:.{prec}f}".format(s[i], prec=decimal_cutoff)
            if len(v) + 1 > column_width[i]:
                column_width[i] = len(v) + 1 + padding
    if line_adaptive:
        max_width = indents + 1
        for i in column_width.keys():
            max_width += column_width[i]
        line_width = max_width
    if printing_file:
        print_func("{}\n".format(line_char * line_width))
        print_func("{}:\n".format(name))
    else:
        print_func("{}".format(line_char * line_width))
        print_func("{}:".format(name))
    for s in to_print:
        row_str = leading_char * indents + " "
        for i, x in enumerate(s):
            v = str(x)
            if isinstance(x, float):
                v = "{:.{prec}f}".format(x, prec=decimal_cutoff)
            if i == 0:
                v = v + ":"
                row_str = row_str + "{:{width}}".format(v, width=column_width[i])
            else:
                row_str = row_str + "{:{width}}".format(v, width=column_width[i])
        if printing_file:
            row_str = row_str + "\n"
        print_func(row_str)
    if printing_file:
        print_func("{}\n".format(line_char * line_width))
    else:
        print_func("{}".format(line_char * line_width))


def createID(first=None, last=None, fullname=None):
    """
    Function to create the id
    :param first: first name
    :param last: last name
    :param fullname: full name, will override first and last name
    :return: the id
    """
    if fullname:
        name = fullname
    else:
        if not first:
            name = last
        elif not last:
            name = first
        else:
            name = first + " " + last
    name = unidecode.unidecode(unescape(name.lower()))
    name = remove_punct_ids.sub("", name)
    return name.replace(" ", "-")


def convertPaperToSortable(p, year_only=False):
    """
    Convert the paper id to a sortable object based on year and id
    :param p: the paper id
    :param year_only: only get the year
    :return: int of form year+paper number or year
    """
    try:
        venue, pnumber = p.split("-")
    except Exception as e:
        print(p)
        raise e
    venue = venue[1:]
    if int(venue) > 60:
        venue = "19" + venue
    else:
        venue = "20" + venue
    if year_only:
        return int(venue)
    return int(venue + pnumber)


def nameFromDict(d):
    """
    Get the string name from a dict
    :param d: dict that has keys "first" and "last"
    :return: the name
    """
    if not d["first"] and not d["last"]:
        return None
    elif not d["first"]:
        return d["last"]
    elif not d["last"]:
        return d["first"]
    else:
        return d["first"] + " " + d["last"]


def getChildText(e, delimiter=""):
    """
    Gets all text of an element
    :param e: lxml element
    :param delimiter: separate the text by
    :return: child text of e separated by delimiter
    """
    return remove_html.sub(delimiter, etree.tostring(e).decode("utf-8")).strip()


def cleanName(n, replace_punct=True):
    """
    Cleans the string of any special characters
    :param n: string
    :param replace_punct: If you want to replace punctuation as well
    :return: cleaned n
    """
    if replace_punct:
        return remove_punct_ids.sub("", unidecode.unidecode(unescape(n))).replace("-", " ")
    else:
        return unidecode.unidecode(unescape(n))


def chunks(l, n):
    """
    Create chunked data
    # TODO: Source needed
    :param l: iterable
    :param n: size of each chunk
    :return: generator of l chunked in l//n + 1 if len(l) % n > 0 chunks of size n
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ncr(n, r):
    """
    n choose r
    # TODO: SOURCE NEEDED
    :param n: int n
    :param r: int r
    :return: n choose r
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def createLogger(logger_name, out_file, msg_format, console_level, file_level=logging.DEBUG):
    """
    Creates the logger
    :param logger_name: name of the logger
    :param out_file: file for the logger to output too
    :param msg_format: Log message format
    :param console_level: Logging level of the console
    :param file_level: Logging level of the file
    :return: The logger object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(out_file)
    fh.setLevel(file_level)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    formatter = logging.Formatter(msg_format)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# TODO: Change the arguments to allow easier input
def printLogToConsole(console_level, msg, level, print_func=print, logger=None, specific_level=None):
    """
    Somewhat redundant function, but I used it a lot just to make printing strings somewhat easier. It DOES NOT send any
    message to the logger
    :param console_level: the logging level of the console log handler
    :param msg: the message to send
    :param level: the level of the message
    :param print_func: function to print with
    :param logger: Logger to print with, Defaults to None for instances you don't want to log as well. For
    example, if you are using a progress bar and you don't want logger messages messing it up
    :param specific_level: If you would rather use FATAL/CRITICAL/Custom string instead of ERROR or WARN/Custom
    string instead of WARNING, specify this
    :return: None
    """
    if level == logging.INFO:
        level_str = "INFO"
    elif level == logging.DEBUG:
        level_str = "DEBUG"
    elif level == logging.WARN:
        level_str = "WARNING"
    elif level == logging.WARNING:
        level_str = "WARNING"
    elif level == logging.ERROR:
        level_str = "ERROR"
    else:
        raise ValueError("Invalid level passed")

    specified_level = False
    if (level_str == "WARNING" or level_str == "ERROR") and specific_level is not None:
        level_str = specific_level
        specified_level = True

    # Checks if the level you passed is INFO or lower, used to prevent duplicate messages
    if level >= 20 and console_level > 20:
        print_func("{}: {}".format(level_str, msg))
        if logger is not None:
            if level_str == "FATAL":
                logger.fatal("{}".format(msg))
            elif level_str == "CRITICAL":
                logger.critical("{}".format(msg))
            elif level_str == "WARN":
                logger.warn("{}".format(msg))
            elif specified_level:
                logger.log("{}:{}".format(level_str, msg))
            else:
                logger.log(level, "{}".format(msg))


def calculatePairStats(pairs, data_keys, metrics=None, logger=None, logger_level=None, special_cases=None,
                       include_special_rest=True, output_dir=None):
    """
    Calculate stats for compared pairs
    :param pairs: the compared pairs
    :param data_keys: The keys that are from CompareAuthors.compare_terms
    :param metrics: metrics you want to return, it is a tuple in the form ("name of metric",metric function). Please
    note,
    the metric function is expected to run on a list of either floats or ints
    :param logger: the logger to log too
    :param logger_level: the level you want to log too
    :param special_cases: Any special cases you want to get the stats of
    :param include_special_rest: Include the special cases in the other pairs stats
    :param output_dir: output directory to write stats too
    :return: None
    """
    if metrics is None:
        metrics = [
            ("avg", np.mean),
            ("median", np.median)
        ]
    if special_cases is None:
        special_cases = []

    def getStats(d):
        key_stats = {x: [] for x in data_keys}
        pbar = tqdm(total=len(d), file=sys.stdout)
        for i in d:
            for j, v in enumerate(i):
                try:
                    key_stats[data_keys[j]].append(v)

                except Exception as e:
                    pbar.close()
                    print(j)
                    raise e
            pbar.update()
        pbar.close()
        out = {}
        for k in key_stats.keys():
            calculated_dict = {}
            for name, algorithm in metrics:
                calculated_dict[name] = algorithm(key_stats[k])
            out[k] = calculated_dict
        return out

    same = []
    different = []
    special_same = []
    special_different = []
    print("INFO: Parsing pairs")
    parse_pbar = tqdm(total=len(pairs), file=sys.stdout)
    for key, tag, data in pairs:
        p1, a, p2, b = key.split()
        is_special_case = False
        for case in special_cases:
            tmp_a = "-".join(a.split("-")[:len(case.split("-"))])
            tmp_b = "-".join(b.split("-")[:len(case.split("-"))])
            if tmp_a == case and tmp_b == case:
                is_special_case = True
                break
        if tag == 1:
            if is_special_case:
                special_same.append(data)
                if include_special_rest:
                    same.append(data)
            else:
                same.append(data)
        else:
            if is_special_case:
                special_different.append(data)
                if include_special_rest:
                    different.append(data)
            else:
                different.append(data)
        parse_pbar.update()
    parse_pbar.close()

    if logger:
        logger.log(logger_level, "{} same pairs".format(len(same)))
        logger.log(logger_level, "{} different pairs".format(len(different)))
        logger.log(logger_level, "{} special same pairs".format(len(special_same)))
        logger.log(logger_level, "{} special different pairs".format(len(special_different)))

    print("INFO: Getting Stats")
    stats = {}
    keys_and_pairs = [("same", same), ("different", different), ("special same", special_same), ("special different",
                                                                                                 special_different)]
    for k, pair_stats in keys_and_pairs:
        stats[k] = getStats(pair_stats)

    def singleMetrics(a):
        args_print_stats = []
        pbar = tqdm(total=len(data_keys) * len(metrics), file=sys.stdout)
        for i in data_keys:
            s = stats[a]
            for name, _ in metrics:
                s_value = s[i][name]
                if not s_value:
                    s_value = 0
                stat_str = "{} {}".format(i, name)
                args_print_stats.append([stat_str, abs(s_value)])
                pbar.update()
        pbar.close()
        return args_print_stats

    def differenceStats(a, b):
        args_print_stats = []
        pbar = tqdm(total=len(data_keys) * len(metrics), file=sys.stdout)
        for i in data_keys:
            s = stats[a]
            d = stats[b]
            for name, _ in metrics:
                s_value = s[i][name]
                if not s_value:
                    s_value = 0
                d_value = d[i][name]
                if not d_value:
                    d_value = 0
                stat_str = "{} {}".format(i, name)
                args_print_stats.append([stat_str, abs(s_value - d_value)])
                pbar.update()
        pbar.close()
        return args_print_stats

    # print("INFO: Getting stats for same")
    # same_stats = singleMetrics("same")
    # print("INFO: Getting stats for different")
    # different_stats = singleMetrics("different")
    print("INFO: Getting stats for same vs different")
    same_different = differenceStats("same", "different")
    print("INFO: Getting stats for special same vs special different")
    special_same_and_different = differenceStats("special same", "special different")
    if output_dir:
        with open(output_dir + "/same_different.txt", "w") as f:

            printStats("Same vs Different", same_different, print_func=f.write, printing_file=True)
        with open(output_dir + "/special_same_special_different.txt", "w") as f:

            printStats("Special Same vs Special Different", special_same_and_different, print_func=f.write,
                       printing_file=True)
    else:
        # printStats("Same", same_stats)
        # printStats("Different", different_stats)
        printStats("Same vs Different", same_different)
        printStats("Special Same vs Special Different", special_same_and_different)


def makeParameterName(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def removeDupes(l):
    return list(set(l))


def createCLIGroup(arguments, group_name, group_description, arg_dict):
    group = arguments.add_argument_group(group_name, group_description)
    for k, v in arg_dict.items():
        default_value, description = v
        if isinstance(default_value, bool):
            group.add_argument("--{}".format(k), nargs="?", const=not default_value, type=type(default_value),
                               default=None, help=description)
        else:
            group.add_argument("--{}".format(k), nargs=1, type=type(default_value), default=None, help=description)


def parseCLIArgs(args, config_handler, debug_mode=False):
    args_passed = {}
    override = False
    save = False
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg == "debug":
            if arg_value:
                args_passed["console_log_level"] = logging.DEBUG
                args_passed["DEBUG_MODE"] = arg_value
        elif arg == "ext_dir":
            if arg_value:
                args_passed["ext_directory"] = True
        elif arg == "overwrite_config":
            override = arg_value
        elif arg == "save_config":
            save = arg_value
        else:
            if arg_value is not None:
                if isinstance(arg_value,list):
                    args_passed[arg] = arg_value[0]
                else:
                    args_passed[arg] = arg_value
    for k, v in args_passed.items():
        if debug_mode:
            print("{}={}".format(k, v))
        else:
            config_handler.addArgument(k, v, override)
    if debug_mode:
        return
    if save:
        config_handler.save()
    return config_handler


def loadData(to_load, logger, config_handler, other_files=None, override_keys=None):
    if not other_files:
        other_files = []
    if not override_keys:
        override_keys = {}
    out = {}
    for file in to_load:
        try:
            path = config_handler[file]
        except KeyError:
            printLogToConsole(config_handler.console_log_level,
                              "{} is not in config_handler, skipping".format(file), logging.WARNING,
                              logger=logger)
            continue
        if not isinstance(path, str):
            printLogToConsole(config_handler.console_log_level,
                              "config_handler[{}] is not a string, skipping".format(file), logging.WARNING,
                              logger=logger)
            continue
        logger.debug("path={}".format(path))
        extension = path.split(".")[-1]
        printLogToConsole(config_handler.console_log_level, "Loading {}".format(file), logging.INFO, logger=logger)
        if "corpus" in file:
            logger.debug("File is a corpus")
            out[file] = [[stemmer.stem(w) for w in x.strip().split()] for x in open(path).readlines()]
        elif extension == "json":
            logger.debug("File has json extension")

            out[file] = ujson.load(open(path))
        elif extension == "txt":
            logger.debug("File has txt extension")
            out[file] = [line.strip() for line in open(path).readlines()]
        elif extension == "csv":
            logger.debug("File has csv extension")
            out[file] = [line.strip().split(",") for line in open(path).readlines()]
        else:
            printLogToConsole(config_handler.console_log_level,
                              "{} is an unknown extension, out[{}] is the io reader result from open".format(extension,
                                                                                                             file),
                              logging.INFO,
                              logger=logger)
            out[file] = open(path)

    logger.debug("Opening other files passed")
    for path in other_files:
        file, extension = path.split("/")[-1].split(".")
        printLogToConsole(config_handler.console_log_level, "Loading {}".format(file), logging.INFO, logger=logger)
        if extension == "json":
            logger.debug("File has json extension")
            out[file] = ujson.load(path)
        elif extension == "txt":
            logger.debug("File has txt extension")
            out[file] = [line.strip() for line in open(path).readlines()]
        elif extension == "csv":
            logger.debug("File has csv extension")
            out[file] = [line.strip().split(",") for line in open(path).readlines()]
        else:
            printLogToConsole(config_handler.console_log_level,
                              "{} is an unknown extension, out[{}] is the io reader result from open".format(extension,
                                                                                                             file),
                              logging.WARNING,
                              logger=logger)
            out[file] = open(path)
    logger.debug("Overriding keys")
    for k,n in override_keys.items():
        logger.debug("Changing {} to {}".format(k,n))
        out[n] = deepcopy(out[k])
        del out[k]
    return out
