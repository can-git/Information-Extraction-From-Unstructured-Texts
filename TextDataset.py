import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import Properties as p
from nltk.stem import PorterStemmer


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, df, train):
        tokenizer = BertTokenizer.from_pretrained(p.BERT_NAME)
        ls = []
        if train:
            # for rows in df.values:
            #     for i, cols in enumerate(rows[1:5]):
            #         if cols == 1:
            #             ls.append(i)
            ls = df.iloc[:, 1:5].values
        else:
            ls = np.random.randint(4, size=len(df))

        self.labels = ls

        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['pathology_report_text']]

        self.ids = [text for text in df['pateint_id']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_ids(self, idx):
        # Fetch a batch of inputs
        return self.ids[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_ids = self.get_batch_ids(idx)

        return batch_ids, batch_texts, batch_y