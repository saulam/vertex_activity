import torch.nn as nn

from .bert import BERT

class VAModel(nn.Module):
    """
    BERT-based Vertex Activity Model
    """

    def __init__(self, bert: BERT):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.bert = bert
        self.decoder_signal = Decoder(self.bert.hidden, 2, activation=False)
        self.decoder_charge = Decoder(self.bert.hidden, 1, activation=True)

    def forward(self, x, noise):
        x = self.bert(x, noise)
        return self.decoder_signal(x), self.decoder_charge(x).view(x.shape[0], -1)

class Decoder(nn.Module):

    def __init__(self, hidden, outsize, activation):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, outsize)
        if activation:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x):
        x = self.linear(x)
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        return x