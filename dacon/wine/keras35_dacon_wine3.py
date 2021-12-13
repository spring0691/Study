import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using \ Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True, choices=['rf','erf'], help="Type of classifier \ to use; can be either 'rf' or 'efr'")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifer_type

# 데이터로드

path = "../_data/dacon/wine/"

train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

