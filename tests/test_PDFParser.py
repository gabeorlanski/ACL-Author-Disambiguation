from unittest import TestCase
import os
import lxml
from io import StringIO
import yaml
import json
import re
from lxml import etree
from collections import defaultdict, Counter
import fuzzysearch
import unidecode
from html import unescape
from tqdm import tqdm
import logging
from copy import deepcopy
from src.utility_functions import *
from src.pdf_parser import PDFParser
from src.paper import Paper
import warnings
import sys


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestPDFParser(TestCase):
    @ignore_warnings
    def setUp(self):
        self.config = json.load(open("/home/gabe/Desktop/research-main/config.json"))
        test_paper_path = os.getcwd() + "/tests/pdfParserTests/"
        data_path = os.getcwd() + "/data"

        self.test_paper1_root = etree.XML(open(test_paper_path + "test_1.tei.xml", "rb").read())
        self.test_paper1_xml = open(test_paper_path + "test_1.tei.xml", "rb").read()
        self.test1_key = "Q13-1004"
        self.test_paper2_root = etree.XML(open(test_paper_path + "test_2.tei.xml", "rb").read())
        self.test_paper2_xml = open(test_paper_path + "test_2.tei.xml", "rb").read()
        self.test2_key = "W19-4450"
        self.test_paper3_root = etree.XML(open(test_paper_path + "test_3.tei.xml", "rb").read())
        self.test_paper4_root = etree.XML(open(test_paper_path + "test_4.tei.xml", "rb").read())

        self.aliases = json.load(open(data_path + "/json/aliases.json"))
        self.papers = json.load(open(data_path + "/json/acl_papers.json"))
        self.id_to_name = json.load(open(data_path + "/json/id_to_name.json"))
        self.same_names = [x.strip() for x in open(data_path + "/txt/same_names.txt").readlines()]

    @ignore_warnings
    def test_getOrganizations(self):
        parser = PDFParser(self.config["parsed pdf path"])
        a = parser.getOrganizations(self.test_paper1_root)

        self.assertEqual(len(a), 1)
        self.assertTrue("aff0" in a)
        a_aff = a["aff0"]
        self.assertDictEqual(a_aff['address'], {})
        self.assertEqual(a_aff["id"], "the-university-of-texas-at-dallas")
        self.assertDictEqual(a_aff["info"], {
            'department': ['Computer Science Department'],
            'institution': ['The University of Texas at Dallas']
        })
        self.assertEqual(a_aff["type"], ["institution"])
        b = parser.getOrganizations(self.test_paper2_root)
        # print(b)
        # self.assertEqual(len(b),2)
        self.assertTrue("aff0" in b)
        self.assertTrue("aff1" in b)
        b_0 = b["aff0"]
        b_1 = b["aff1"]

        self.assertDictEqual(b_0['address'], {})
        self.assertEqual(b_0["id"], "university-of-washington")
        self.assertEqual(b_0["type"], ["institution"])
        self.assertDictEqual(b_0["info"], {
            'department': ['Department of Electrical and Computer Engineering'],
            'institution': ['University of Washington']
        })
        self.assertDictEqual(b_1['address'], {})
        self.assertEqual(b_1["id"], "laix-inc")
        self.assertEqual(b_1["type"], ["laboratory"])
        self.assertDictEqual(b_1["info"], {
            'department': [],
            'laboratory': ['LAIX Inc']
        })

    @ignore_warnings
    def test_matchNames(self):
        parser = PDFParser(self.config["parsed pdf path"])
        test_a = [
            [("john-smith", "john smith"), ("william-west", "william west"), ("robert-john", "robert john")],
            [("john-smith", "john smith"), ("will-west", "will west"), ("robert-j-john", "robert j jogn")]
        ]
        matched, unknown = parser._matchAuthors(*test_a)
        self.assertDictEqual(matched, {
            ("john-smith", "john smith"): "john-smith",
            ("will-west", "will west"): "william-west",
            ("robert-j-john", "robert j jogn"): "robert-john"
        })
        self.assertEqual(len(unknown), 0)

    @ignore_warnings
    def test_parseAuthors(self):
        parser = PDFParser(self.config["parsed pdf path"])
        parser.loadData(self.papers, self.aliases, self.id_to_name, self.same_names)
        test_a = {
            "yang-liu-icsi": "Yang Liu",
            "xian-qian": "Xian Qian"
        }
        test_b = {
            "yang-liu-icsi": "Yang Liu",
            "mari-ostendorf": "Mari Ostendorf",
            "farah-nadeem": "Farah Nadeem",
            "huy-nguyen": "Huy Nguyen"
        }
        test_c = {
            "yang-liu-icsi": "Yang Liu",
            "mari-ostendorf": "Mari Ostendorf",
            "yang-liu-uw": "Yang Liu",
            "huy-nguyen": "Huy Nguyen"
        }

        out, manual_fixes_required, unknown, fixed_count, correct_with_manual, errors = parser._parseAuthors(test_a,
                                                                                                             self.test_paper1_root,
                                                                                                             "", {})
        self.assertDictEqual(out, {
            'xian-qian': {
                'email': None,
                'aff_key': 'aff0'
            },
            'yang-liu-icsi': {
                'email': 'yangl@hlt.utdallas.edu',
                'aff_key': 'aff0'
            }
        })
        self.assertEqual(manual_fixes_required, [])
        self.assertEqual(unknown, [])
        self.assertEqual(fixed_count, 1)
        self.assertEqual(correct_with_manual, 0)

        out, manual_fixes_required, unknown, fixed_count, correct_with_manual, errors = parser._parseAuthors(test_b,
                                                                                                             self.test_paper2_root,
                                                                                                             "", {})
        self.assertDictEqual(out, {
            "yang-liu-icsi": {
                'email': "yang.liu@liulishuo.com",
                'aff_key': 'aff1'
            },
            "mari-ostendorf": {
                'email': "ostendor@uw.edu",
                'aff_key': 'aff0'
            },
            "farah-nadeem": {
                'email': "farahn@uw.edu",
                'aff_key': 'aff0'
            },
            "huy-nguyen": {
                'email': "huy.nguyen@liulishuo.com",
                'aff_key': 'aff1'
            }
        })
        self.assertEqual(len(manual_fixes_required), 0)
        self.assertEqual(unknown, [])
        self.assertEqual(fixed_count, 1)
        self.assertEqual(correct_with_manual, 0)

        out, manual_fixes_required, unknown, fixed_count, correct_with_manual, errors = parser._parseAuthors(test_c,
                                                                                                             self.test_paper3_root,
                                                                                                             "", {})
        self.assertDictEqual(out, {
            "mari-ostendorf": {
                'email': "ostendor@uw.edu",
                'aff_key': 'aff0'
            },
            "huy-nguyen": {
                'email': "huy.nguyen@liulishuo.com",
                'aff_key': 'aff1'
            }
        })
        self.assertEqual(len(manual_fixes_required), 2)
        self.assertEqual(unknown, [])
        self.assertEqual(fixed_count, 0)
        self.assertEqual(correct_with_manual, 0)

        test_c_manual = {
            ('yang-liu', 'aff0', 'farahn@uw.edu'): "yang-liu-uw",
            ('yang-liu', 'aff1', 'yang.liu@liulishuo.com'): "yang-liu-icsi"
        }
        out, manual_fixes_required, unknown, fixed_count, correct_with_manual, errors = parser._parseAuthors(test_c,
                                                                                                             self.test_paper3_root,
                                                                                                             "",
                                                                                                             test_c_manual)
        self.assertDictEqual(out, {
            "mari-ostendorf": {
                'email': "ostendor@uw.edu",
                'aff_key': 'aff0'
            },
            "huy-nguyen": {
                'email': "huy.nguyen@liulishuo.com",
                'aff_key': 'aff1'
            },
            "yang-liu-icsi": {
                'email': 'yang.liu@liulishuo.com',
                'aff_key': 'aff1'
            },
            "yang-liu-uw": {
                'email': 'farahn@uw.edu',
                'aff_key': 'aff0'
            }
        })
        self.assertEqual(len(manual_fixes_required), 0)
        self.assertEqual(unknown, [])
        self.assertEqual(fixed_count, 0)
        self.assertEqual(correct_with_manual, 2)

    @ignore_warnings
    def test_parsePaper(self):
        parser = PDFParser(self.config["parsed pdf path"])
        parser.loadData(self.papers, self.aliases, self.id_to_name, self.same_names)
        a = parser.papers[self.test1_key]
        rtr, status, errors = parser._parsePaper(a, self.test_paper1_xml, {})
        a_res, manual_fixes = rtr
        a_aff = {
            'xian-qian': {
                'email': None,
                'affiliation': {
                    'address': {},
                    'info': {
                        'department': ['Computer Science Department'],
                        'institution': ['The University of Texas at Dallas']
                    },
                    'id': 'the-university-of-texas-at-dallas',
                    'type': ['institution']
                }
            },
            'yang-liu-icsi': {
                'email': 'yangl@hlt.utdallas.edu',
                'affiliation': {
                    'address': {},
                    'info': {
                        'department': ['Computer Science Department'],
                        'institution': ['The University of Texas at Dallas']
                    },
                    'id': 'the-university-of-texas-at-dallas',
                    'type': ['institution']
                }
            }
        }
        self.assertEqual(status, 0)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(manual_fixes), 0)
        self.assertEqual(a_res.pid, a.pid)
        self.assertEqual(a_res.title, a.title)
        self.assertEqual(a_res.abstract, a.abstract)
        self.assertDictEqual(a_res.authors, a.authors)
        self.assertEqual(a_res.unknown, a.unknown)
        self.assertDictEqual(a_res.affiliations, a_aff)

        b = parser.papers[self.test2_key]
        rtr, status, errors = parser._parsePaper(b, self.test_paper2_xml, {})
        b_res, manual_fixes = rtr
        b_aff = {
            "farah-nadeem": {
                'email': "farahn@uw.edu",
                'affiliation': {
                    'address': {},
                    'info': {
                        'department': ['Department of Electrical and Computer Engineering'],
                        'institution': [
                            'University of Washington']

                    },
                    'id': 'university-of-washington',
                    'type': ['institution']
                }
            },
            "huy-nguyen": {
                'email': "huy.nguyen@liulishuo.com",
                'affiliation': {
                    'address': {},
                    'info': {
                        'laboratory': ['LAIX Inc'],
                        'department': []

                    },
                    'id': 'laix-inc',
                    'type': ['laboratory']

                }
            },
            "mari-ostendorf": {
                'email': "ostendor@uw.edu",
                'affiliation': {
                    'address': {},
                    'info': {
                        'department': ['Department of Electrical and Computer Engineering'],
                        'institution': [
                            'University of Washington']

                    },
                    'id': 'university-of-washington',
                    'type': ['institution']
                }
            },
            'yang-liu-icsi': {
                'email': 'yang.liu@liulishuo.com',
                'affiliation': {
                    'address': {},
                    'info': {
                        'laboratory': ['LAIX Inc'],
                        'department': []

                    },
                    'id': 'laix-inc',
                    'type': ['laboratory']
                }
            },
        }
        self.assertEqual(status, 0)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(manual_fixes), 0)
        self.assertEqual(b_res.pid, b.pid)
        self.assertEqual(b_res.title, b.title)
        self.assertEqual(b_res.abstract, b.abstract)
        self.assertDictEqual(b_res.authors, b.authors)
        self.assertEqual(b_res.unknown, b.unknown)
        for auth, info in b_res.affiliations.items():
            self.assertTrue(auth in b_aff)
            self.assertEqual(info["email"], b_aff[auth]["email"])
            self.assertEqual("affiliation" in info, "affiliation" in b_aff[auth])
            res_aff = info["affiliation"]
            true_aff = b_aff[auth]["affiliation"]
            for k in res_aff.keys():
                self.assertTrue(k in true_aff)
                if k == "info":
                    self.assertDictEqual(dict(res_aff[k]), true_aff[k])
                else:
                    self.assertEqual(res_aff[k], true_aff[k])

    # @ignore_warnings
    # def test_parallelCall(self):
    #     parser = PDFParser(self.config["parsed pdf path"], cores=2, in_parallel=True)
    #     parser.loadData(self.papers, self.aliases, self.id_to_name, self.same_names)
    #     parser(os.getcwd() + "/data/pdf_xml", 200)
