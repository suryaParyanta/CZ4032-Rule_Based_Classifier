import os
import argparse
import time

import pandas as pd

from sklearn.model_selection import GridSearchCV

from classifier.cba_classifier import CBA
from classifier.cmar_classifier import CMAR


def parse_argument():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--model",
        help = "Model used for training.",
        choices = ["CBA", "CMAR"],
        default= "CBA",
        type = str
    )

    args.add_argument(
        "--dataset",
        help = "Dataset used for training.",
        choices =  ["iris", "wine", "titanic", "breast_cancer"],
        default = "iris",
        type = str
    )

    return args.parse_args()


def get_parameters(model_type:str, feature_names:list = None):
    """
    Get list of hyperparameters for grid search.

    :param model_type:    CBA or CMAR
    :param feature_names: List of feature names
    """
    if model_type == "CBA":
        num_combs = 10
        parameters = {
            "feature_names": [feature_names],
            "support_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "conf_threshold": [0.5],
            "method": ["M2"], # can change to M1, the accuracy is almost the same
            "max_candidates": [80000]
        }
        
    elif model_type == "CMAR":
        num_combs = 10 * 7
        parameters = {
            "feature_names": [feature_names],
            "support_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "conf_threshold": [0.5],
            "database_coverage": [1, 2, 3, 4, 5, 6, 7],
        }
    return parameters, num_combs


def get_dataset(dataset_name):
    data_dir = os.path.join("datasets", "processed")
    return  pd.read_csv(os.path.join(data_dir, f"{dataset_name}_processed.csv"))


def eval_cv(args):
    """
    Evaluate model using 10-fold cross validation. We will also perform grid search
    to get the best performance of each model.
    """
    dataset = get_dataset(args.dataset)
    X = dataset.drop(["class"], axis = 1)
    y = dataset["class"]
    feature_names = X.columns
    X, y = X.to_numpy(), y.to_numpy()

    if args.model == "CBA":
        estimator = CBA()
    elif args.model == "CMAR":
        estimator = CMAR()
    
    parameters, num_combs = get_parameters(args.model, feature_names = feature_names)

    start_time = time.time()
    gs = GridSearchCV(estimator, parameters, scoring = "accuracy", cv = 10)
    gs.fit(X, y)
    end_time = time.time()
    
    # results
    print("Best Parameters:")
    for k, v in gs.best_params_.items():
        if k != "feature_names":
            print(f"    {k}:  {v}")
    print("\nBest 10-Fold CV Accuracy: ", round(gs.best_score_, 4))
    print(f"Average run time:  {round((end_time - start_time) / num_combs, 2)} seconds\n")


if __name__ == "__main__":
    args = parse_argument()
    print(f"\n{args.model} evaluation on {args.dataset} dataset")
    eval_cv(args)