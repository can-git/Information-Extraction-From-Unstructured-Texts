import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import transformers
from transformers import BertTokenizer
from PIL import Image
import Properties as p
import nltk
from BertClassifier import BertClassifier
from InitializationLayer import InitializationLayer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

grades_csv = pd.read_csv(p.DEFAULT_PATH_TRAIN)
reports_tsv = pd.read_csv(p.DEFAULT_PATH_DATA, delimiter="\t", encoding="ISO-8859-1")
df = pd.merge(grades_csv, reports_tsv, on="pateint_id")
test_patients = pd.read_csv(p.DEFAULT_PATH_TEST)
df_test = pd.merge(test_patients, reports_tsv, on="pateint_id")


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'G1': 0,
          'G2': 1,
          'G3': 2,
          'G4': 3
          }


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, df, train):
        ls = []
        if train:
            for rows in df.values:
                for i, cols in enumerate(rows[1:5]):
                    if cols == 1:
                        ls.append(i)
        else:
            ls = np.random.randint(4, size=len(df))
        self.labels = ls
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['pathology_report_text']]

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

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


df_train, df_val = train_test_split(df, test_size=0.1, random_state=1)


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = TextDataset(train_data, True), TextDataset(val_data, True)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=p.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=p.BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in train_dataloader:
            train_label = torch.randint(0, 2, (p.BATCH_SIZE,))
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # output = model(input_id)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


EPOCHS = p.EPOCHS
# model = BertClassifier()
model = InitializationLayer()
LR = p.LR

train(model, df_train, df_val, LR, EPOCHS)


def evaluate(model, test_data):
    test = TextDataset(test_data, False)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # print(output[0])



            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    # new_df = pd.DataFrame({"img_id": names, "cancer_score": predictions})
    # new_df.to_csv("can_yilmaz_assignment_1.csv", index=False)

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')



evaluate(model, df_test)

