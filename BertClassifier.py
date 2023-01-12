import torch
from torch import nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(768, 4)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256*2, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):

        # _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # dropout_output = self.dropout(pooled_output)
        # lstm_output = self.lstm(pooled_output)

        # hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        # linear_output = self.linear(hidden.view(-1, 256*2))
        # linear_output = self.linear(lstm_output[:, -1])
        # final_layer = self.softmax(linear_output)


        return self.softmax(output)

        # return final_layer