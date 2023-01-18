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

    def replace(self, df):
        column_name = 'pathology_report_text'
        replace_prob = 0.1
        tokenizer = BertTokenizer.from_pretrained(p.BERT_NAME)
        vocab = tokenizer.get_vocab()
        # Iterate over the rows of the dataframe
        for i, row in df.iterrows():
            # Tokenize the text in the selected column
            tokenized_text = tokenizer.tokenize(row[column_name])

            # Iterate over the tokenized text and randomly replace tokens
            for i, token in enumerate(tokenized_text):
                if np.random.rand() < replace_prob:
                    # Replace the token with a random token from the vocabulary
                    tokenized_text[i] = random.choice(list(vocab.keys()))

            # Convert the tokenized text back to a string
            modified_text = tokenizer.convert_tokens_to_string(tokenized_text)

            # Update the value of the selected column with the modified text
            df.at[i, column_name] = modified_text

        return df

    def getItem(self):
        # return self.df, self.df_test,
        return self.df_train, self.df_val, self.df_test,
