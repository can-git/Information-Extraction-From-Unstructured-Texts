import re
import string
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import Properties as p


class Preprocess():
    def __init__(self):
        grades_csv = pd.read_csv(p.DEFAULT_PATH_TRAIN)
        reports_tsv = pd.read_csv(p.DEFAULT_PATH_DATA, delimiter="\t", encoding="ISO-8859-1")
        df = pd.merge(grades_csv, reports_tsv, on="pateint_id")
        test_patients = pd.read_csv(p.DEFAULT_PATH_TEST)

        df = self.transform(df)

        self.df_test = pd.merge(test_patients, reports_tsv, on="pateint_id")

        self.labels = {'G1': 0,
                       'G2': 1,
                       'G3': 2,
                       'G4': 3
                       }

        self.df_train, self.df_val = train_test_split(df, test_size=0.1, random_state=1)

    def transform(self, df):

        punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        df["pathology_report_text"] = df["pathology_report_text"].apply(lambda x: re.sub(f"[{punctuation}]", "", x))

        return df

    def getItem(self):
        return self.df_train, self.df_val, self.df_test
