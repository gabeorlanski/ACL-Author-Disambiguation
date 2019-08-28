from unittest import TestCase
from src.dense_neural_net import DenseNN
import numpy as np


class TestDenseNN(TestCase):
    def test__parseData(self):
        test_data = [
            ("P1 yang-liu-ict P2 yang-liu-icsi", 0, [1, 1]),
            ("P1 yang-liu-ict P3 bang-liu", 0, [2, 2]),
            ("P1 yang-liu-ict P4 yang-liu-ict", 1, [3, 3]),
            ("P1 bang-liu P4 bang-liu", 1, [4, 4]),

        ]
        dnn = DenseNN( special_cases=["yang-liu"])
        same, different, special_same, special_different = dnn._parseData(test_data)
        self.assertEqual(same, [([4, 4], 1)])
        self.assertEqual(different, [([2, 2], 0)])
        self.assertEqual(special_same, [([3, 3], 1)])
        self.assertEqual(special_different, [([1, 1], 0)])

    def test_convertData(self):
        test = [
            ([1, 1, 1, 1], 0),
            ([0, 0, 0, 0], 0),
            ([0, 0, 0, 1], 1),
        ]
        correct_X = np.asarray([np.asarray(x[0]) for x in test])
        correct_Y = np.asarray([np.asarray(x[1]) for x in test])
        X, Y = DenseNN._convertData(test)
        np.testing.assert_array_equal(X, correct_X)
        np.testing.assert_array_equal(Y, correct_Y)

    def test_splitTrainTest(self):
        test_same = [([1,1,1,1],1) for x in range(32)]
        test_diff = [([0,0,0,0],0) for x in range(48)]
        dnn = DenseNN(special_cases=["yang-liu"])
        train,test = dnn._splitTrainTest(test_same, test_diff)
        self.assertEqual(len(train),70)
        self.assertEqual(len(test),10)

        same_count = 0
        diff_count = 0
        for i in train:
            if i[1] == 1:
                same_count +=1
            else:
                diff_count+=1
        self.assertEqual(same_count,28)
        self.assertEqual(diff_count,42)
        self.assertEqual(same_count+diff_count,70)

        same_count = 0
        diff_count = 0
        for i in test:
            if i[1] == 1:
                same_count += 1
            else:
                diff_count += 1
        self.assertEqual(same_count, 4)
        self.assertEqual(diff_count, 6)
        self.assertEqual(same_count+diff_count,10)