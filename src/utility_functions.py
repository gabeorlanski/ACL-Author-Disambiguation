from collections import defaultdict
import unidecode
from html import unescape
import re
from lxml import etree

remove_punct_ids = re.compile("[^\w\s-]")
remove_html = re.compile("<[^>]*>")
remove_punct = re.compile("[^\w\s]")


def printStats(name, to_print, leading_char="-", indents=2, decimal_cutoff=3, line_char="=", line_width=20,
               line_adaptive=False, padding=2, default_width=6):
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
    print("{}".format(line_char * line_width))
    print("{}:".format(name))
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
        print(row_str)
    print("{}".format(line_char * line_width))


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


def convertPaperToSortable(p):
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
