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
from nltk import pos_tag
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
import random


class Preprocess:
    def __init__(self):
        grades_csv = pd.read_csv(p.DEFAULT_PATH_TRAIN)
        reports_tsv = pd.read_csv(p.DEFAULT_PATH_DATA, delimiter="\t", encoding="ISO-8859-1")

        df = pd.merge(grades_csv, reports_tsv, on="pateint_id")
        df = df.drop_duplicates(subset=["pateint_id"])
        test_patients = pd.read_csv(p.DEFAULT_PATH_TEST)

        self.df = self.transform(df)

        self.df_test = pd.merge(test_patients, reports_tsv, on="pateint_id")

        self.labels = {'G1': 0,
                       'G2': 1,
                       'G3': 2,
                       'G4': 3
                       }

        self.df_train, self.df_val = train_test_split(df, test_size=0.25, random_state=56)
        # self.df_val, self.df_test2 = train_test_split(self.df_val, test_size=0.5, random_state=1)

    def transform(self, df):
        nltk.download('stopwords')
        stemmer = PorterStemmer()
        punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        stop = stopwords.words('english')

        # df = self.replace(df)

        df["pathology_report_text"] = df["pathology_report_text"].apply(
            lambda x: re.sub(f"[{punctuation}]", " ", x))
        df['pathology_report_text'] = df['pathology_report_text'].apply(
            lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
        df['pathology_report_text'] = df['pathology_report_text'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        return df


    def getItem(self):
        # return self.df, self.df_test,
        return self.df_train, self.df_val, self.df_test,
