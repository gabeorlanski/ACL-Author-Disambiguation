import pickle
from src.vote_classifier import VoteClassifier
from src.config_handler import ConfigHandler
from src.utility_functions import createCLIGroup, createCLIShared, parseCLIArgs
import gc
import os
import argparse
import json

arguments = argparse.ArgumentParser(
    description="Train the model using data from preprocess_data.py. You can specify these in config.json instead of using "
                "command line arguments",
    formatter_class=argparse.MetavarTypeHelpFormatter)
createCLIShared(arguments)
createCLIGroup(arguments, "VoteClassifier",
               "Arguments for the VoteClassifier, check the documentation of VoteClassifier to see default "
               "values",
               VoteClassifier.parameters)

if __name__ == '__main__':
    gc.collect()
    args = arguments.parse_args()
    log_path = os.getcwd() + '/logs/train.log'
    with open(log_path, 'w'):
        pass
    print("INFO: Starting Preprocess Data")
    gc.collect()
    config_raw = json.load(open("config.json"))
    config = ConfigHandler(config_raw, "train", raise_error_unknown=True)
    config = parseCLIArgs(args, config)
    data = pickle.load(open(config["tagged_pairs"], "rb"))
    scores = []
    weights = {
        "Nearest Neighbors": 1,
        "Decision Tree": 3,
        "Random Forest": 2,
        "Neural Net": 2,
        "Naive Bayes": 1,
        "AdaBoost": 2,
        "QDA": 1,
    }
    config.addArgument("classifier_weights", weights)
    # params = {
    #     'Nearest Neighbors': {
    #         'algorithm': 'ball_tree',
    #         'leaf_size': 10,
    #         'metric': 'manhattan',
    #         'n_neighbors': 9,
    #         'p': 1,
    #         'weights': 'uniform'
    #     },
    #     'Decision Tree': {
    #         'max_depth': 5,
    #         'max_features': 5,
    #         'min_samples_leaf': 4,
    #         'min_samples_split': 2,
    #         'min_weight_fraction_leaf': 0.0
    #     },
    #     'Random Forest': {
    #         'max_depth': 5,
    #         'max_features': 5,
    #         'min_samples_leaf': 5,
    #         'min_samples_split': 2,
    #         'min_weight_fraction_leaf': 0.0,
    #         'warm_start': False
    #     },
    #     'AdaBoost': {
    #         'algorithm': 'SAMME',
    #         'learning_rate': 1.0,
    #         'n_estimators': 100
    #     },
    #     'Naive Bayes': {
    #     },
    #     'QDA': {
    #         # 'store_covariance': True,
    #         'tol': 0.25
    #     },
    #     'Neural Net': {
    #         'activation': 'identity',
    #         'alpha': 0.14285714285714285,
    #         'learning_rate': 'adaptive',
    #         'power_t': 0.6,
    #         'solver': 'adam',
    #         'tol': 0.3333333333333333
    #     }
    # }
    params = {
        'Nearest Neighbors': {
            'n_neighbors': 7,
        },
        'Decision Tree': {
            'max_depth': 5,
        },
        'Random Forest': {
            'max_depth': 5,
            'max_features': 5,
            'n_estimators': 10
        },
        'AdaBoost': {},
        'Naive Bayes': {},
        'QDA': {},
        'Neural Net': {
            'alpha': 1,
            'max_iter': 1000
        }
    }
    classifiers = [
        ("Nearest Neighbors", "KNeighborsClassifier"),
        ("Decision Tree", "DecisionTreeClassifier"),
        ("Random Forest", "RandomForestClassifier"),
        ("AdaBoost", "AdaBoostClassifier"),
        ("Naive Bayes", "GaussianNB"),
        ("QDA", "QuadraticDiscriminantAnalysis"),
        ("Neural Net", "MLPClassifier")
    ]

    special_keys = [x.strip() for x in open(config["test_special_keys"]).readlines() if x != "\n"]
    config.addArgument("special_cases", special_keys)
    vote_classifier = VoteClassifier(data, classifiers=classifiers, **config["VoteClassifier"])
    vote_classifier.createModel(params)
    vote_classifier.trainModel()
    vote_classifier.evaluate()
    vote_classifier.save()
