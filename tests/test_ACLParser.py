import unittest
from src.acl_parser import *
import os
import json


class TestACLParser(unittest.TestCase):

    def test_parseNameVariants(self):
        config = json.load(open("config.json"))
        parser = ACLParser(config["ACLParserXpaths"])
        a_id = "jose-i-abreu"
        a_name = {
            "first": "José I.",
            "last": "Abreu"
        }
        a_variants = ["jose i. abreu", "josé abreu"]
        b_id = "yang-liu-ict"
        b_name = {
            "first": "Yang",
            "last": "Liu"
        }
        c_id = "james-allan"
        c_name = {
            "first": "James",
            "last": "Allan"
        }
        c_similar = ["james-allen"]

        parser.parseNameVariants("/home/gabe/Desktop/research-main/data")
        self.assertTrue(a_id in parser.id_to_name)
        self.assertDictEqual(a_name, parser.id_to_name[a_id])
        for name in a_variants:
            self.assertEqual(a_id, parser.aliases[name])

        self.assertTrue(b_id in parser.id_to_name)
        self.assertDictEqual(b_name, parser.id_to_name[b_id])
        self.assertTrue("Yang Liu" in parser.same_name)

        self.assertTrue(c_id in parser.id_to_name)
        self.assertDictEqual(c_name, parser.id_to_name[c_id])
        self.assertEqual(c_similar, parser.similar_names[c_id])
        self.assertEqual([c_id], parser.similar_names[c_similar[0]])

    def test_parseACL(self):
        config = json.load(open("config.json"))
        parser = ACLParser(config["ACLParserXpaths"])
        parser("/home/gabe/Desktop/research-main/data/xml/ACL/", "/home/gabe/Desktop/research-main/data")

        test_conflict = "eugenio-martinez-camara"
        test_conflict_one = "W17-0908"
        test_paper_1 = "C02-2007"
        test_paper_1_data = {
                "yang-liu-ict": "Yang Liu",
                "shiwen-yu": "Shiwen Yu",
                "jiangsheng-yu": "Jiangsheng Yu"
            }
        test_paper_2 = "C14-1179"
        test_paper_2_data = {
                "yang-liu-ict":"Yang Liu",
                "maosong-sun":"Maosong Sun",
                "dakun-zhang":"Dakun Zhang",
                "peng-li":"Peng Li",
                "tatsuya-izuha":"Tatsuya Izuha"
            }

        self.assertEqual(parser.conflicts[test_conflict], [{'first': 'Eugenio', 'last': 'Martinez Camara'}, {'first': 'Eugenio', 'last': 'Martinez Camara'}])
        self.assertTrue(test_conflict in parser.papers[test_conflict_one].authors)

        self.assertDictEqual(parser.papers[test_paper_1].authors,test_paper_1_data)
        self.assertDictEqual(parser.papers[test_paper_2].authors, test_paper_2_data)


