import math
import torch
from torch import nn
from transformers import BertModel
import torch_geometric.nn as gnn


class InitializationLayer(nn.Module):
    def __init__(self):
        super(InitializationLayer, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, bidirectional=True)

        # pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)

        # fully connected layer
        self.fc = nn.Linear(128 * 2, 4)

        # softmax activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # pass input through pre-trained BERT
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)

        # pass pooled output through bi-directional LSTM
        lstm_output, _ = self.bilstm(pooled_output)

        lstm_output = lstm_output.unsqueeze(2)

        # apply pooling
        pooled_output = self.pool(lstm_output)

        # add the batch dimension
        # pooled_output = pooled_output.squeeze()
        pooled_output = pooled_output.view(-1, 128 * 2)


        out = self.fc(pooled_output)

        # apply softmax activation
        out = self.softmax(out)

        return out
