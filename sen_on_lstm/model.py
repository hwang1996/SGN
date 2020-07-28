import torch
import torch.nn as nn

from locked_dropout import LockedDropout
from ON_LSTM import ONLSTMStack
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, chunk_size, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM'], 'RNN type is not supported'
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )

        # self.decoder = nn.Linear(ninp, ntoken)

        self.prob = nn.Linear(1, 15)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     #if nhid != ninp:
        #     #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.embedding = nn.Linear(768, 400)
        self.drop_out = nn.Dropout(p=dropoute)
        # self.linear = nn.Linear(400, ntoken)
        
        self.sen_out = nn.Sequential(
            nn.Conv1d(10, 5, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(5, 1, 3, stride=1, padding=1),
        )
        self.result = nn.Sequential(
            nn.Conv1d(19, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, 3, stride=1, padding=1),
        )
        self.word_rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )


    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, cand_ids, hidden, hidd, hidd_cand):
        ## suppose batch_size = 80
        inputs_ = inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)).transpose(0, 1)  #[80, 15, 10] -> [10, 1200]
        cand_ids_ = cand_ids.view(cand_ids.size(0)*cand_ids.size(1), cand_ids.size(2)).transpose(0, 1)   #[80, 4, 10] -> [10, 320]

        emb = embedded_dropout(
            self.encoder, inputs_,
            dropout=self.dropoute if self.training else 0
        )
        emb = self.lockdrop(emb, self.dropouti)   #[10, 1200, 400]
        emb_out, _, _, _, _ = self.word_rnn(emb, hidd)     #[10, 1200, 400]
        sen_emb = self.sen_out(emb_out.permute(1, 0, 2))  #[1200, 1, 400]
        sen_emb = sen_emb.view(inputs.size(0), inputs.size(1), -1).transpose(0, 1) #[80, 15, 400] -> [15, 80, 400]

        cand_emb = embedded_dropout(
            self.encoder, cand_ids_,
            dropout=self.dropoute if self.training else 0
        )
        cand_emb = self.lockdrop(cand_emb, self.dropouti)   #[10, 320, 400]
        cand_emb_out, _, _, _, _ = self.word_rnn(cand_emb, hidd_cand)     #[10, 320, 400]
        cand_sen_emb = self.sen_out(cand_emb_out.permute(1, 0, 2))   #[320, 1, 400]
        cand_sen_emb = cand_sen_emb.view(cand_ids.size(0), cand_ids.size(1), -1).transpose(0, 1)  #[4, 80, 400]       

        ##1. language modeling

        # raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
        # self.distance = distances

        # output = self.lockdrop(raw_output, self.dropout)
        # result = output.view(output.size(0)*output.size(1), output.size(2))
        # result_prob = self.decoder(result)

        ##2. quick thought
        raw_output, hidden, raw_outputs, outputs, distances = self.rnn(sen_emb, hidden)
        self.distance = distances

        output = self.lockdrop(raw_output, self.dropout)

        output = output.permute(1, 0, 2)
        result = self.result(output)
        cand_scores = torch.matmul(result, cand_sen_emb.permute(1, 2, 0)).squeeze(1)

        return result, cand_scores, hidden, raw_outputs, outputs, cand_emb

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X