#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:25:14 2017

@author: chadliamino
"""
import argparse
import warnings

#import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

#from sklearn import metrics

random_seed = 1
test_split = 0.3
data = "kddcup99.csv"
normal = "normal"

def main():
    args = _init_args()
    model = _init_model(args)
    x, y = _load_data(args)
    x_enc, y_enc = _encode_data(x, y)
    x_train, _x_test, y_train, _y_test = _split_data(x_enc, y_enc, args)
    _train(model, x_train, y_train, args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="nb",
        choices=["nb", "tree", "forest"])
    p.add_argument(
        "--tree-max-depth", type=int, default=None)
    p.add_argument(
        "--tree-min-samples-split", type=int, default=2)
    return p.parse_args()

def _init_model(args):
    if args.model == "nb":
        print("Using Gaussian NB classifier")
        return GaussianNB()
    elif args.model == "tree":
        print("Using decision tree classifier")
        return DecisionTreeClassifier(
            max_depth=args.tree_max_depth,
            min_samples_split=args.tree_min_samples_split,
        )
    elif args.model == "forest":
        print("Using random forest classifier")
        return RandomForestClassifier()
    else:
        assert False, model

def _load_data(_args):
    print("Loading data")
    with open(data) as f:
        f.readline()
        dataset = pd.read_csv(f, header=None)
    cols = dataset.columns.tolist()
    dataset[cols[-1]] = dataset[cols[-1]].apply(
        lambda x: "normal" if x.startswith("normal") else "attack")

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 41].values

    return x, y

def _encode_data(x, y):
    print("Encoding data")

    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()

    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        onehotencoder_1 = OneHotEncoder(categorical_features=[1])
        x = onehotencoder_1.fit_transform(x).toarray()

        onehotencoder_2 = OneHotEncoder(categorical_features=[4])
        x = onehotencoder_2.fit_transform(x).toarray()

        onehotencoder_3 = OneHotEncoder(categorical_features=[70])
        x = onehotencoder_3.fit_transform(x).toarray()

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    return x, y

def _split_data(x, y, _args):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_split,
        random_state=random_seed)

    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    return x_train, x_test, y_train, y_test

def _train(model, x_train, y_train, args):
    print("Cross validating model")

    scores = cross_validate(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=5,
        scoring=["accuracy", "precision", "recall", "f1"])

    print("Accuracy: %f" % scores["test_accuracy"].mean())
    print("Precision: %f" % scores["test_precision"].mean())
    print("Recall: %f" % scores["test_recall"].mean())
    print("F-1: %f" % scores["test_f1"].mean())

if __name__ == "__main__":
    main()
