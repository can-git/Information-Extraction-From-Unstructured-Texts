import math
import torch
from torch import nn
from transformers import BertModel


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
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        # pass input through pre-trained BERT
        _, out = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.dropout(out)
        # pass pooled output through bi-directional LSTM
        out, _ = self.bilstm(out)

        out = out.unsqueeze(2)

        # apply pooling
        out = self.pool(out)

        # add the batch dimension
        # pooled_output = pooled_output.squeeze()
        out = out.view(-1, 128 * 2)

        out = self.fc(out)

        # apply softmax activation
        out = self.softmax(out)

        return out
