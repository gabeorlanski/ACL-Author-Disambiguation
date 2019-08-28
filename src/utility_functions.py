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

remove_punct_ids = re.compile("[^\w\s-]")
remove_html = re.compile("<[^>]*>")
remove_punct = re.compile("[^\w\s]")


def printStats(name, to_print, leading_char="-", indents=2, decimal_cutoff=3, line_char="=", line_width=20,
               line_adaptive=False, padding=2, default_width=6, print_func=None, printing_file=False):
    if not print_func:
        print_func=print
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
    if not d["first"]:
        return d["last"]
    elif not d["last"]:
        return d["first"]
    else:
        return d["first"] + " " + d["last"]


def getChildText(e, delimiter=""):
    return remove_html.sub(delimiter, etree.tostring(e).decode("utf-8")).strip()


def cleanName(n, replace_punct=True):
    if replace_punct:
        return remove_punct_ids.sub("", unidecode.unidecode(unescape(n))).replace("-", " ")
    else:
        return unidecode.unidecode(unescape(n))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def createLogger(logger_name, out_file, msg_format, console_level, file_level=logging.DEBUG):
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


def printLogToConsole(console_level, msg, level):
    if level == logging.INFO:
        level_str = "INFO"
    elif level == logging.DEBUG:
        level_str = "DEBUG"
    elif level == logging.WARN:
        level_str = "WARN"
    elif level == logging.WARNING:
        level_str = "WARNING"
    elif level == logging.ERROR:
        level_str = "ERROR"
    elif level == logging.CRITICAL:
        level_str = "CRITICAL"
    elif level == logging.FATAL:
        level_str = "FATAL"
    else:
        raise ValueError("Invalid level passed")
    if level >= 20 and console_level > 20:
        print("{}: {}".format(level_str, msg))


def calculatePairStats(pairs, data_keys, metrics=None, logger=None, logger_level=None, special_cases=None,
                       include_special_rest=True, return_results=False, output_dir=None):
    if metrics is None:
        metrics = [
            ("avg", np.mean),
            ("median", np.median)
        ]
    if special_cases is None:
        special_cases = []

    def getStats(d):
        key_stats = {x: [] for x in data_keys}
        pbar = tqdm(total=len(d),file=sys.stdout)
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
            for name,algorithm in metrics:
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
        pbar = tqdm(total=len(data_keys)*len(metrics),file=sys.stdout)
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

    def differenceStats(a,b):
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
                stat_str = "{} {}".format(i,name)
                args_print_stats.append([stat_str,abs(s_value-d_value)])
                pbar.update()
        pbar.close()
        return args_print_stats
    # print("INFO: Getting stats for same")
    # same_stats = singleMetrics("same")
    # print("INFO: Getting stats for different")
    # different_stats = singleMetrics("different")
    print("INFO: Getting stats for same vs different")
    same_different = differenceStats("same","different")
    print("INFO: Getting stats for special same vs special different")
    special_same_and_different = differenceStats("special same","special different")
    if output_dir:
        with open(output_dir+"/same_different.txt","w") as f:

            printStats("Same vs Different",same_different,print_func=f.write,printing_file=True)
        with open(output_dir+"/special_same_special_different.txt","w") as f:

            printStats("Special Same vs Special Different",special_same_and_different,print_func=f.write,printing_file=True)
    else:
        # printStats("Same", same_stats)
        # printStats("Different", different_stats)
        printStats("Same vs Different", same_different)
        printStats("Special Same vs Special Different",special_same_and_different)


def get_name(parameters):
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