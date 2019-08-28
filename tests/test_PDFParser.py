from unittest import TestCase
import os
import lxml
from io import StringIO
import yaml
import json
import re
from lxml import etree
from collections import Counter
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
        if "/tests" in os.getcwd():
            os.chdir("..")
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
        papers_tmp = json.load(open(data_path + "/json/acl_papers.json"))
        self.papers ={x:Paper(**v) for x,v in papers_tmp.items()}
        self.id_to_name = json.load(open(data_path + "/json/id_to_name.json"))
        self.same_names = [x.strip() for x in open(data_path + "/txt/same_names.txt").readlines()]
        self.default_args = {
            "aliases":self.aliases,
            "papers":self.papers,
            "id_to_name":self.id_to_name,
            "same_names":self.same_names,
            "sim_cutoff":.75
        }

    @ignore_warnings
    def test_getOrganizations(self):
        parser = PDFParser(**self.default_args)
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
        parser = PDFParser(**self.default_args)
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
        parser = PDFParser(**self.default_args)
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
        parser = PDFParser(**self.default_args)
        a = parser.papers[self.test1_key]
        a_args = a, self.test_paper1_xml, {}
        rtr, status, errors = parser(a_args)
        a_res, manual_fixes,_,_ = rtr
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
        b_args = b, self.test_paper2_xml, {}
        rtr, status, errors = parser(b_args)
        b_res, manual_fixes,_,_ = rtr
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

    @ignore_warnings
    def test_parseCitations(self):
        parser = PDFParser(**self.default_args)
        a = parser.papers[self.test1_key]
        test_1 = [
            {
                "title":"An additive algorithm for solving linear programs with zero-one variables",
                "type":"a",
                "authors":[
                    "Egon Balas"
                ],
                "pub_type":"j",
                "pub_title":"Operations Research",
                "volume":None,
                "issue":4,
                "date":1965,
                "date_type":"published",
            },
            {
                "title":"The best of bothworlds -a graph-based completion model for transition-based parsers",
                "type":"a",
                "authors":[
                    "Bernd Bohnet",
                    "Jonas Kuhn"
                ],
                "pub_type":"m",
                "pub_title":"Proc. of EACL",
                "volume":None,
                "issue":None,
                "date":"2012",
                "date_type":"published",
            },
            {
                "title": "A transitionbased system for joint part-of-speech tagging and labeled non-projective dependency parsing",
                "type":"a",
                "authors":[
                    "Bernd Bohnet",
                    "Joakim Nivre"
                ],
                "pub_type":"m",
                "pub_title":"Proc. of EMNLP-CoNLL",
                "volume":None,
                "issue":None,
                "date":2012,
                "date_type":"published",
            },
            {
                "title":"Projected gradient methods for linearly constrained problems",
                "type":"a",
                "authors":[
                    "Paul Calamai",
                    "Jorge More"
                    ],
                "pub_type":"j",
                "pub_title":"Mathematical Programming",
                "volume":None,
                "issue":1,
                "date":1987,
                "date_type":"published",
            },
            {
                "title":"Experiments with a higher-order projective dependency parser",
                "type":"a",
                "authors":[
                    "Xavier Carreras"
                ],
                "pub_type":"m",
                "pub_title":"Proc. of EMNLPCoNLL",
                "volume":None,
                "issue":None,
                "date":2007,
                "date_type":"published",
            },
            {
                "title":"Coarse-tofine n-best parsing and maxent discriminative reranking",
                "type":"a",
                "authors":[
                    "Eugene Charniak",
                    "Mark Johnson"
                ],
                "pub_type":"m",
                "pub_title":"Proc. of ACL",
                "volume":None,
                "issue":None,
                "date":2005,
                "date_type":"published",
            },
            {
                "title":"Utilizing dependency language models for graph-based dependency parsing models",
                "type":"a",
                "authors":[
                    "Wenliang Chen",
                    "Min Zhang",
                    "Haizhou Li"
                ],
                "pub_type":"m",
                "pub_title":"Proc. of ACL",
                "volume":None,
                "issue":None,
                "date":"2012",
                "date_type":"published",
            },
        ]
        test_titles = [x["title"] for x in test_1]
        res, errors = parser._parseCitations("test1",self.test_paper1_root)
        for i in res:
            self.assertTrue("title" in i)
            self.assertTrue("type" in i)
            self.assertTrue("authors" in i)
            self.assertTrue("pub_type" in i)
            self.assertTrue("pub_title" in i)
            self.assertTrue("volume" in i)
            self.assertTrue("issue" in i)
            self.assertTrue("date" in i)
            self.assertTrue("date_type" in i)
            if i["title"] in test_titles:
                test_index = test_titles.index(i["title"])
                self.assertDictEqual(i,test_1[test_index])

    @ignore_warnings
    def test_parseSections(self):
        parser = PDFParser(**self.default_args)
        a = parser.papers[self.test1_key]
        expected = {
            "1":{
                "title":"Introduction"
            },
            "2":{
                "title":"Graph Based Parsing",
                "1":"Problem Definition",
                "2":"Dynamic Programming for Local Models",
            },
            "3":{
                "title":"The Proposed Method",
                "1":"Basic Idea",
                "2":"The Upper Bound Function",
                "3":"Branch and Bound Based Parsing",
                "4":"Lower Bound Initialization",
                "5":"Summary"
            },
            "4":{
                "title":"Experiments",
                "1":"Experimental Settings",
                "2":"Baseline DP Based Second Order Parser",
                "3":"BB Based Parser with Non local Features",
                "4":"Implementation Details",
                "5":"Main Result",
                "6":"Tradeoff Between Accuracy and Speed"
            },
            "5":{
                "title":"Discussion",
                "1":"Polynomial Non local Factors",
                "2":"k Best Parsing",
            },
            "6":{
                "title":"Conclusion"
            }

        }
        missing =[]
        incorrect = []
        res, rv = parser._parseSections(self.test_paper1_root)
        self.assertEqual(rv,0)
        for k,info in expected.items():
            if k not in res:
                missing.append(k)
                continue
            res_k = res[k]
            if len(res_k) != len(info):
                incorrect.append([k+" len",len(res_k),len(info)])
                continue
            for t in info.keys():
                if t not in res_k:
                    missing.append(k+"."+t)
                    continue
                expected_val = info[t]
                actual_val = res_k[t]
                if expected_val != actual_val:
                    incorrect.append([k+"."+t,actual_val,expected_val])
                    continue

        self.assertEqual(missing,[])
        self.assertEqual(incorrect,[])