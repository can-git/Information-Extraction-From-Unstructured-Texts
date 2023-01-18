import torch
from torch import nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence
import Properties as p


class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()
        self.hidden_size = 64

        self.bert = BertModel.from_pretrained(p.BERT_NAME)
        self.lstm = nn.LSTM(768, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.hidden_size*2, 4)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        # self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        # self.fc4 = nn.Linear(self.hidden_size//4, 4)

        self.pool = nn.MaxPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):

        _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.dropout(x)
        x = self.pool(x)
        x, _ = self.lstm(x)

        x = x.unsqueeze(2)
        x = x.view(-1, self.hidden_size*2)

        x = self.fc(x)
        return self.softmax(x)
