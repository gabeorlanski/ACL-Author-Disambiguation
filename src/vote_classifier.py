import numpy as np
import pickle
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random
import time
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import logging
from src.utility_functions import createLogger, printLogToConsole
from tqdm import tqdm
import json


class VoteClassifier:
    parameters = dict(
        classifier_weights=[{},"Weights for each classifier"],
        test_fraction=[8,"Amount to split data into train and test. ex test_fraction=8 train=7/8 * data, test=1/7*data"],
        model_save_path=['/models/',"Path where to save models"],
        model_name=["VC1","Model Name"],
        special_cases=[[],"Special cases you want to evaluate"],
        rand_seed=[1,"Random seed"],
        cutoff=[1000,"Amount of data cases to use"],
        special_only=[False,"Train and test only on special cases"],
        diff_same_ratio=[1,"Ratio of diff:same, and vice versa"],
        train_all_estimators=[False,"Train every estimator provided, otherwise -"],
        voting=["hard","Voting types, either soft or hard"],
    )

    def __init__(self, data, classifiers, classifier_weights=None, test_fraction=8, save_data=False,
                 ext_directory=False, save_path=None, model_save_path='/models/', model_name=None, special_cases=None,
                 rand_seed=None, cutoff=None, special_only=False, console_log_level=logging.ERROR,
                 file_log_level=logging.DEBUG, log_format=None, log_path=None, diff_same_ratio=1,
                 train_all_estimators=False, voting="hard"):

        if not log_format:
            log_format = '%(asctime)s|%(levelname)8s|%(module)20s|%(funcName)20s: %(message)s'
        if not log_path:
            log_path = os.getcwd() + "/logs/dnn.log"
        if not save_path:
            if ext_directory:
                self.save_path = os.getcwd() + "/data/pickle"
            else:
                self.save_path = os.getcwd() + "/data"
        else:
            self.save_path = save_path
        self.logger = createLogger("DNN", log_path, log_format, console_log_level, file_log_level)
        self.console_log_level = console_log_level
        self.test_fraction = test_fraction
        if special_cases is None:
            special_cases = []
        self.special_cases = special_cases
        self.rand_seed = rand_seed
        if not self.rand_seed:
            self.rand_seed = random.randint(0, 9999)
            random.seed(self.rand_seed)

        self.logger.debug("seed is {}".format(self.rand_seed))
        self.cutoff = cutoff
        self.dif_same_ratio = diff_same_ratio
        self.save_data = save_data
        self.special_only = special_only
        self.model_save_path = model_save_path
        self.model_name = model_name
        if not model_name:
            self.model_name = "VC1"
        if not classifier_weights:
            classifier_weights = {}
            for n, m in classifiers:
                classifiers[n] = 1
        if len(classifier_weights) != len(classifiers):
            self.logger.error("len(classifier_weights)={}".format(len(classifier_weights)))
            self.logger.error("len(classifier)={}".format(len(classifiers)))
            raise ValueError("Classifier weights must be the same length as the number of classifiers passed")
        self.classifier_weights = classifier_weights
        self.classifiers = classifiers
        self.estimators = []
        self.model = None
        self.train_all_estimators = train_all_estimators
        self.classifier_params = {}
        self.voting = voting
        self.train, self.test, self.special_test, self.special_train = self._createTrainTest(data)

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
                    is_special_case = True
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
        same = np.array([x[0] for x in same])
        diff = np.array([x[0] for x in diff])
        same_larger = False
        if len_same > len_diff:
            same_larger = True
            pair_count = int(len_diff * self.dif_same_ratio)
        else:
            pair_count = int(len_same * self.dif_same_ratio)
        if same_larger:
            out_diff = diff
            out_same = resample(same, n_samples=pair_count, random_state=self.rand_seed)
        else:
            out_same = same
            out_diff = resample(diff, n_samples=pair_count, random_state=self.rand_seed)
        return out_same, out_diff

    @staticmethod
    def convertToUsable(data):
        return [(x[2], x[1]) for x in data]

    def _splitTrainTest(self, same, diff, special_same=None, special_diff=None):

        self.logger.debug("Creating train and test")
        Y = []
        X = np.concatenate((same, diff))
        self.logger.debug("# of same = {}".format(same.shape[0]))
        for i in range(same.shape[0]):
            Y.append(1)
        self.logger.debug("# of different = {}".format(diff.shape[0]))
        for i in range(diff.shape[0]):
            Y.append(0)
        if special_same is not None:
            X = np.concatenate((X, special_same))
            self.logger.debug("# of special same = {}".format(special_same.shape[0]))
            for i in range(special_same.shape[0]):
                Y.append(1)
        if special_diff is not None:
            X = np.concatenate((X, special_diff))
            self.logger.debug("# of special diff= {}".format(special_diff.shape[0]))
            for i in range(special_diff.shape[0]):
                Y.append(0)
        Y = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / self.test_fraction,
                                                            random_state=self.rand_seed)
        return {
                   "X": X_train,
                   "Y": Y_train
               }, {
                   "X": X_test,
                   "Y": Y_test
               }

    def _createTrainTest(self, data):
        printLogToConsole(self.console_log_level, "Creating train and test", logging.INFO)
        self.logger.info("Creating train and test")
        same, different, special_same, special_different = self._parseData(data)

        def saveData(d, file_path):
            with open(file_path, "wb") as f:
                to_save = [x for x in d]
                pickle.dump(to_save, f)

        if self.save_data:
            saveData(same, self.save_path + "/same.pickle")
            saveData(different, self.save_path + "/different.pickle")
            saveData(special_same, self.save_path + "/special_same.pickle")
            saveData(special_different, self.save_path + "/special_different.pickle")
            all_pairs = []
            for k, t, _ in [*same, *different, *special_same, *special_different]:
                all_pairs.append([k, t])
            with open(self.save_path + "/save_pairs.pickle", "wb") as f:
                pickle.dump(all_pairs, f)

        same = self.convertToUsable(same)
        different = self.convertToUsable(different)
        special_same = self.convertToUsable(special_same)
        special_different = self.convertToUsable(special_different)
        same, different = self._selectPairsToUse(same, different)
        special_same, special_different = self._selectPairsToUse(special_same, special_different)
        train, test = self._splitTrainTest(same, different, special_same, special_different)
        special_train, special_test = self._splitTrainTest(special_same, special_different)

        printLogToConsole(self.console_log_level, "Splitting non-special pairs", logging.INFO)
        self.logger.info("Splitting non-special pairs")

        printLogToConsole(self.console_log_level, "Splitting special pairs", logging.INFO)
        self.logger.info("Splitting special pairs")

        return train, test, special_train, special_test

    def createModel(self, classifier_parameters):
        self.classifier_params = classifier_parameters
        printLogToConsole(self.console_log_level, "Creating model", logging.INFO)
        self.logger.info("Creating model")
        self.logger.debug("{} estimators".format(len(self.classifiers)))
        weights = []
        for n, m in self.classifiers:
            self.logger.debug("n={}".format(n))
            self.logger.debug("m={}".format(m))
            weights.append(self.classifier_weights[n])
            if n not in classifier_parameters:
                self.logger.error("{} is not in classifier_parameters".format(n))
                raise KeyError("{} is not in classifier_parameters".format(n))
            if m == "GaussianNB":
                self.estimators.append((n, GaussianNB(**classifier_parameters[n])))
            elif m == "KNeighborsClassifier":
                self.estimators.append((n, KNeighborsClassifier(**classifier_parameters[n])))
            elif m == "MLPClassifier":
                self.estimators.append((n, MLPClassifier(**classifier_parameters[n])))
            elif m == "SVC":
                self.estimators.append((n, SVC(**classifier_parameters[n])))
            elif m == "RBF":
                self.estimators.append((n, RBF(**classifier_parameters[n])))
            elif m == "RandomForestClassifier":
                self.estimators.append((n, RandomForestClassifier(**classifier_parameters[n])))
            elif m == "AdaBoostClassifier":
                self.estimators.append((n, AdaBoostClassifier(**classifier_parameters[n])))
            elif m == "QuadraticDiscriminantAnalysis":
                self.estimators.append((n, QuadraticDiscriminantAnalysis(**classifier_parameters[n])))
            elif m == "DecisionTreeClassifier":
                self.estimators.append((n, DecisionTreeClassifier(**classifier_parameters[n])))
            elif m == "GaussianProcessClassifier":
                self.estimators.append((n, GaussianProcessClassifier(**classifier_parameters[n])))
            else:
                self.logger.error("Unknown classifier")
                raise ValueError("{} is not a supported classifier".format(m))

        self.model = VotingClassifier(self.estimators, voting=self.voting, weights=weights)

    def trainModel(self, voting="hard"):
        printLogToConsole(self.console_log_level, "Training model", logging.INFO)
        self.logger.info("Training model")

        if self.special_only:
            self.logger.debug("Training on special data")
            self.logger.debug("Test cases: {}".format(len(self.special_train["X"])))
            train = self.special_train
        else:
            self.logger.debug("Training on all data")
            self.logger.debug("Test cases: {}".format(len(self.train["X"])))
            train = self.train

        X = train["X"]
        Y = train["Y"]
        if self.train_all_estimators:
            for n, m in self.estimators:
                t0 = time.time()
                m.fit(X, Y)
                t1 = time.time()
                progress_str = "Finished fitting classifier {} in {:.2f}s".format(n, t1 - t0)
                printLogToConsole(self.console_log_level, progress_str, logging.INFO)
                self.logger.info(progress_str)
        printLogToConsole(self.console_log_level, "Fitting the VotingClassifier Model", logging.INFO)
        self.logger.info("Fitting the VotingClassifier Model")
        self.model.fit(X, Y)
        printLogToConsole(self.console_log_level, "Finished fitting model", logging.INFO)
        self.logger.info("Finished fitting model")

    def evaluate(self):
        printLogToConsole(self.console_log_level, "Evaluating model", logging.INFO)
        self.logger.info("Evaluating model")
        if self.train_all_estimators:
            predictions = {}
            special_predictions = {}
            for n, m in self.estimators:
                self.logger.debug("Making predictions for {}".format(n))
                predictions[n] = m.predict(self.test["X"])
                special_predictions[n] = m.predict(self.special_test["X"])

            printLogToConsole(self.console_log_level, "Results for all estimators", logging.INFO)
            self.logger.info("Results for all estimators")
            if not self.special_only:
                printLogToConsole(self.console_log_level,
                                  "First stat line is on normal test, second is for special cases",
                                  logging.INFO)
            column_str = "{} {:>11} {:>11} {:>11}".format(" " * 25, "precision", "recall", "f1-score")
            printLogToConsole(self.console_log_level, column_str, logging.INFO)
            self.logger.info(column_str)
            for k, pred in predictions.items():
                precision, recall, _, _ = precision_recall_fscore_support(self.test["Y"], pred, average="binary")
                f1 = f1_score(self.test["Y"], pred, average="binary")
                stat_str = "{:<25} {:>11.2f} {:>11.2f} {:>11.2f}".format(k + ":", precision, recall, f1)
                printLogToConsole(self.console_log_level, stat_str, logging.INFO)
                self.logger.info(stat_str)
                if self.special_only:
                    continue
                precision, recall, _, _ = precision_recall_fscore_support(self.special_test["Y"],
                                                                          special_predictions[k],
                                                                          average="binary")
                f1 = f1_score(self.special_test["Y"], special_predictions[k], average="binary")
                stat_str = "{:<25} {:>11.2f} {:>11.2f} {:>11.2f}".format(k + ":", precision, recall, f1)
                printLogToConsole(self.console_log_level, stat_str, logging.INFO)
                self.logger.info(stat_str)

        model_predictions = self.model.predict(self.test["X"])
        printLogToConsole(self.console_log_level, "Model stats on test data:", logging.INFO)
        self.logger.info("Model stats on test data")
        stats = classification_report(self.test["Y"], model_predictions, target_names=["Different", "Same"])
        print(stats)
        self.logger.info(stats)
        if not self.special_only:
            model_predictions = self.model.predict(self.special_test["X"])
            printLogToConsole(self.console_log_level, "Model stats on special cases data:", logging.INFO)
            self.logger.info("Model stats on special cases data")
            stats = classification_report(self.special_test["Y"], model_predictions, target_names=["Different", "Same"])
            print(stats)
            self.logger.info(stats)

    def save(self):
        path = os.getcwd() + self.model_save_path + self.model_name
        if not os.path.exists(path):
            os.mkdir(path)

        # I got permission denied when using os.path.join
        with open(path + "/model.pickle", "wb") as f:
            pickle.dump(self.model, f)

        parameters_dict = {
            "classifiers": self.classifiers,
            "classifier_weights": self.classifier_weights,
            "classifier_params": self.classifier_params,
            "special_only": self.special_only,
            "test_fraction": self.test_fraction,
            "rand_seed": self.rand_seed,
            "diff_same_ratio": self.dif_same_ratio,
            "cutoff": self.cutoff
        }
        with open(path + "/parameters.json", "w") as f:
            json.dump(parameters_dict, f, indent=4)

        printLogToConsole(self.console_log_level, "Saved model {} to {}".format(self.model_name, path), logging.INFO)
        self.logger.info("Saved model {} to {}".format(self.model_name, path))
