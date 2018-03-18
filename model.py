import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from IPython.core.debugger import Pdb


class AttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, input):
        output = self.mlp(input)
        attn_weight = F.softmax(output.matmul(self.context_vector), dim=0)
        attended_output = attn_weight.matmul(output)
        return attended_output


class AttendedSequenceEmbedding(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50, output_dim=100, bidirectional=True, use_gpu=True, batch_first=True):
        super(AttendedSequenceEmbedding, self).__init__()
        self.use_gpu = use_gpu
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=bidirectional, batch_first=self.batch_first)
        if bidirectional:
            gru_output_dim = 2 * hidden_dim
        else:
            gru_output_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(gru_output_dim, output_dim),
            nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, sequences, seqlens):
        assert(sequences.size() == (seqlens.sum(), self.input_dim))
        padded_inputs = Variable(torch.zeros(len(seqlens), int(seqlens.max()), self.input_dim))
        if self.use_gpu:
            padded_inputs = padded_inputs.cuda()
        begin_idx = 0
        for idx, length in enumerate(seqlens):
            padded_inputs[idx, :length] = sequences[begin_idx:begin_idx + length]
            begin_idx += length
        print(padded_inputs.shape)
        indices = np.argsort(-seqlens).tolist()
        print(indices)
        seqlens = seqlens[indices]
        padded_inputs = padded_inputs[indices]

        packed_input = pack_padded_sequence(padded_inputs, seqlens, batch_first=self.batch_first)
        print(packed_input)
        self.gru.flatten_parameters()
        packed_output, _ = self.gru(packed_input)
        padded_outputs, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # TODO: Check if applying attention after splitting output is faster.
        padded_outputs = self.mlp(padded_outputs)

        # undo sort
        orig_indices = [0] * len(indices)
        for i in range(len(indices)):
            orig_indices[indices[i]] = i
        print(orig_indices)
        padded_outputs = padded_outputs[orig_indices]

        # apply attention
        outputs = [padded_outputs[i, :seqlens[i]] for i in range(padded_outputs.size(0))]
        attn_weights = [F.softmax(output.matmul(self.context_vector), dim=0) for output in outputs]
        attended_outputs = [attn_weight.matmul(output) for attn_weight, output in zip(attn_weights, outputs)]
        attended_outputs = torch.stack(attended_outputs)
        return attended_outputs


class HAN(nn.Module):
    def __init__(self, vocab_size, word_emb_dim=100, gru_hidden_dim=50, emb_dim=100, output_dim=3, batch_size=64, use_gpu=True):
        super(HAN, self).__init__()
        self.use_gpu = use_gpu
        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim, padding_idx=1)
        self.sent_embedding = AttendedSequenceEmbedding(input_dim=word_emb_dim, hidden_dim=gru_hidden_dim,
                                                        output_dim=emb_dim, bidirectional=True)
        self.doc_embedding = AttendedSequenceEmbedding(input_dim=emb_dim, hidden_dim=gru_hidden_dim,
                                                       output_dim=emb_dim, bidirectional=True)
        self.classifier = nn.Linear(emb_dim, output_dim)

    # documents is a list of list of list of words
    def forward(self, documents):
        words = Variable(torch.LongTensor([word for doc in documents for sent in doc for word in sent]))
        if self.use_gpu:
            words = words.cuda()
        word_embeddings = self.word_embedding(words)

        # sent_lengths is a list of list of #words in each sentence in each doc
        sent_lengths = np.array([len(sent) for doc in documents for sent in doc])
        sent_embeddings = self.sent_embedding(word_embeddings, sent_lengths)

        # doc_lengths is a list of #sentences in each doc
        doc_lengths = np.array([len(doc) for doc in documents])
        doc_embeddings = self.doc_embedding(sent_embeddings, doc_lengths)

        outputs = self.classifier(doc_embeddings)
        return F.softmax(outputs, dim=1)
