import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.metrics import BinaryAccuracy
import numpy as np
from tqdm import tqdm
import sys
import os
import random
import gc
from src.utility_functions import createLogger
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from src.compare_authors import CompareAuthors
import pickle

# Credit goes to user pwais for this fix for abseil colliding with python.logging. REMOVE ME WHEN ABSEIL IS UPDATED
try:
    # FIXME(https://github.com/abseil/abseil-py/issues/99)
    # FIXME(https://github.com/abseil/abseil-py/issues/102)
    # Unfortunately, many libraries that include absl (including Tensorflow)
    # will get bitten by double-logging due to absl's incorrect use of
    # the python logging library:
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
    # The code below fixes this double-logging.  FMI see:
    #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493

    import logging

    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass


class DenseNN:
    def __init__(self, epochs=3, layers=4, dropout=.1, activation=tf.keras.activations.relu, test_fraction=8,
                 funnel=2, optimizer='rmsprop', loss='binary_crossentropy', metrics=None,save_path=None, model_save_path='/models/',
                 model_name=None,load_model=None, special_cases=None, rand_seed=None, start_neurons=16, batch_size=10000, cutoff=None,
                 normalize=True, save_pairs=False, special_only=False,console_log_level=logging.ERROR, file_log_level=logging.DEBUG, log_format=None, log_path=None,diff_same_ratio=1):
        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/dnn.log"
        if not save_path:
            self.save_path = os.getcwd()+"/data/pickle"
        else:
            self.save_path = save_path
        self.logger = createLogger("DNN", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level

        if special_cases is None:
            special_cases = []
        if metrics is None:
            metrics = [BinaryAccuracy]

        self.special_cases = special_cases
        self.epochs = epochs
        self.funnel = funnel
        self.layers = layers
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.activation = activation
        self.test_fraction = test_fraction
        if rand_seed:
            random.seed(rand_seed)
        self.start_neurons = start_neurons
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.normalize = normalize
        self.dif_same_ratio = diff_same_ratio
        self.save_pairs = save_pairs
        self.special_only = special_only
        self.model = tf.keras.models.Sequential()
        self.train = None
        self.test = None
        self.special_train =None
        self.special_test = None

    def _parseData(self, data):
        same = []
        different = []
        special_same = []
        special_different = []
        pbar = tqdm(total=len(data), file=sys.stdout)
        for k, t, d in data:
            p1, a, p2, b = k.split()

            special_a = False
            special_b = False
            is_special_case = False
            for case in self.special_cases:
                if case in b:
                    special_a = True

                if case in a:
                    special_b = True
                if special_a and special_b:
                    is_special_case= True
                    break

            if t == 1:
                if is_special_case:
                    special_same.append((k, 1, d))
                else:
                    same.append((k, 1, d))
            else:
                if is_special_case:
                    special_different.append((k, 0, d))
                else:
                    different.append((k, 0, d))
            pbar.update()
        pbar.close()
        if self.cutoff:
            same = random.sample(same, self.cutoff)
            different = random.sample(different, self.cutoff)
        return same, different, special_same, special_different

    def _selectPairsToUse(self, same, diff):
        len_same = len(same)
        len_diff = len(diff)
        if len_same > len_diff:
            pair_count = int(len_diff * self.dif_same_ratio)
        else:
            pair_count = int(len_same * self.dif_same_ratio)

        try:
            out_same = random.sample(same, pair_count)
        except:
            out_same = same[:pair_count]
        try:
            out_diff = random.sample(diff, pair_count)
        except:
            out_diff = diff[:pair_count]
        return out_same, out_diff

    def _convertData(self, data):
        pbar = tqdm(total=len(data), file=sys.stdout)
        X, Y = [], []
        for d, t in data:
            to_add = None
            if isinstance(d, list):
                to_add = np.asarray(d).reshape(1, len(d))
            else:
                to_add = d
            X.append(to_add)
            Y.append(t)
            pbar.update()
        pbar.close()
        try:
            if X[0].shape[0] == 1:
                out = np.asarray(X).reshape(len(X), X[0].shape[1])
            else:
                out = np.asarray(X).reshape(len(X), X[0].shape[0])
            return out, np.asarray(Y)
        except Exception as e:
            print(len(X))
            raise e

    def _splitTrainTest(self, same, different):
        tmp_train = []
        tmp_test = []
        split_same = len(same) // self.test_fraction
        split_diff = len(different) // self.test_fraction

        tmp_train.extend(same[split_same:])
        tmp_test.extend(same[:split_same])

        tmp_train.extend(different[split_diff:])
        tmp_test.extend(different[:split_diff])

        return tmp_train, tmp_test

    def createTrainTest(self, data):
        print("INFO: Parsing data")
        same, different, special_same, special_different = self._parseData(data)
        same, different = self._selectPairsToUse(same, different)
        special_same,special_different = self._selectPairsToUse(special_same,special_different)

        def saveData(d,file_path):
            with open(file_path, "wb") as f:
                to_save = [x for x in d]
                pickle.dump(to_save, f)
        if self.save_pairs:
            saveData(same,self.save_path+"/same.pickle")
            saveData(different,self.save_path+"/different.pickle")
            saveData(special_same,self.save_path+"/special_same.pickle")
            saveData(special_different,self.save_path+"/special_different.pickle")

        same = self.convertToUsable(same)
        different = self.convertToUsable(different)
        special_same = self.convertToUsable(special_same)
        special_different = self.convertToUsable(special_different)
        print("INFO: Splitting normal data")
        train, test = self._splitTrainTest(same, different)
        print("INFO: Splitting special data")
        special_train, special_test = self._splitTrainTest(special_same, special_different)

        train.extend(special_train)
        test.extend(special_test)

        random.shuffle(train)
        random.shuffle(test)
        random.shuffle(special_train)
        random.shuffle(special_test)

        out_train = {}
        out_test = {}
        out_special_train = {}
        out_special_test = {}
        # I needed to collect garbage b/c these datasets are WAY too big to not do that
        print("INFO: Converting data to np arrays")
        out_train["X"], out_train["Y"] = self._convertData(train)
        gc.collect()
        out_test["X"], out_test["Y"] = self._convertData(test)
        gc.collect()
        out_special_train["X"], out_special_train["Y"] = self._convertData(special_train)
        # gc.collect()
        out_special_test["X"], out_special_test["Y"] = self._convertData(special_test)
        gc.collect()
        if self.normalize:
            out_train["X"] = tf.keras.utils.normalize(out_train["X"], axis=1)
            out_test["X"] = tf.keras.utils.normalize(out_test["X"], axis=1)
            out_special_test["X"] = tf.keras.utils.normalize(out_special_test["X"], axis=1)
        return out_train, out_test, out_special_train, out_special_test

    def createModel(self) -> tf.keras.models.Sequential:
        input_shape = self.train["X"].shape[1:]
        model = tf.keras.models.Sequential()
        current_neurons = self.start_neurons
        current_dropout = self.dropout
        for i in range(self.layers - 1):
            model.add(Dense(current_neurons, input_shape=input_shape, activation=self.activation))
            model.add(Dropout(current_dropout))

            if self.funnel:
                current_neurons = current_neurons // self.funnel
                current_dropout = current_dropout // self.funnel
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    @staticmethod
    def convertToUsable(data):
        return [(x[2], x[1]) for x in data]

    def __call__(self, data):
        print("Creating data")
        self.train, self.test, self.special_train, self.special_test = self.createTrainTest(data)
        print("Creating Model")
        self.createModel()

    def train(self):
        print("INFO: Training model...")
        if self.special_only:
            print("INFO: Using only special cases for fitting")
            self.model.fit(self.special_train["X"], self.special_train["Y"], epochs=self.epochs)
        else:
            self.model.fit(self.train["X"], self.train["Y"], epochs=self.epochs)

    def evaluate(self):
        print("INFO: Evaluating on test")
        normal = self.model.evaluate(self.test["X"], self.test["Y"])

        print("INFO: Evaluating on special test")
        special = self.model.evaluate(self.special_test["X"], self.special_test["Y"])
        return normal,special
