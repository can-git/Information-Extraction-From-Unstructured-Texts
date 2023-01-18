import torch
from torch import nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence
import Properties as p


class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(p.BERT_NAME)
        self.dropout = nn.Dropout(0.2)

        self.hidden_size = 32
        self.lstm = nn.LSTM(768, self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):

        _, out = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        out, _ = self.lstm(out)
        out = out.unsqueeze(2)
        out = self.dropout(out)
        out = out.view(-1, self.hidden_size*2)
        out = self.linear(out)
        out = self.softmax(out)
        return out
