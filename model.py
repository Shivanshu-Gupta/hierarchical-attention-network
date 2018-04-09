import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython.core.debugger import Pdb

from utils import log


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


def pack(sequences, seqlens, use_gpu):
    assert(np.count_nonzero(seqlens) == seqlens.shape[0])
    # pad inputs
    padded_inputs = Variable(torch.zeros(seqlens.shape[0], int(seqlens.max()), sequences.size(1)))
    if use_gpu:
        padded_inputs = padded_inputs.cuda()
    begin = 0
    for idx, length in enumerate(seqlens):
        padded_inputs[idx, :length] = sequences[begin:begin + length]
        begin += length
    indices = np.argsort(-seqlens)
    seqlens = seqlens[indices]
    padded_inputs = padded_inputs[torch.LongTensor(indices).cuda()]
    packed_input = pack_padded_sequence(padded_inputs, seqlens, batch_first=True)
    return packed_input, indices


class AttendedSeqEmbedding(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50, output_dim=100,
                 rnn_type='gru', use_gpu=True, batch_first=True):
        super(AttendedSeqEmbedding, self).__init__()
        self.use_gpu = use_gpu
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=batch_first)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=batch_first)
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.rnn_type = rnn_type
        rnn_output_dim = 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(rnn_output_dim, output_dim),
            nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(output_dim))
        self.context_vector.data.normal_(0, 0.1)

    def forward(self, sequences, seqlens):
        packed_input, indices = pack(sequences, seqlens, self.use_gpu)
        # self.rnn.flatten_parameters()
        packed_output, _ = self.rnn(packed_input)
        padded_outputs, sorted_seqlens = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # undo sort
        # padded_outputs = padded_outputs[torch.LongTensor(orig_indices).cuda()]

        # apply attention
        # mlp_input = torch.cat([padded_outputs[i, :seqlens[i]] for i in range(len(seqlens))], dim=0)
        # mlp_output = self.mlp(mlp_input)
        mlp_output = self.mlp(padded_outputs)
        attn_weight = mlp_output.matmul(self.context_vector)
        # end = np.cumsum(seqlens)
        attended_outputs = torch.stack([F.softmax(attn_weight[i, :length], dim=0).matmul(padded_outputs[i, :length]) for i, length in enumerate(sorted_seqlens)], dim=0)

        # undo sort
        orig_indices = [0] * indices.shape[0]
        for i in range(indices.shape[0]):
            orig_indices[indices[i]] = i
        attended_outputs = attended_outputs[torch.LongTensor(orig_indices).cuda()]

        return attended_outputs


class HAN(nn.Module):
    def __version__(self):
        return '2.1.0'

    def __init__(self, review_vocab_size, summary_vocab_size, word_emb_dim=100, rnn_hidden_dim=50, emb_dim=100, output_dim=3, use_summary=True, combined_lookup=False, rnn_type='gru', use_summ_mlp=False, batch_size=64, use_gpu=True):
        super(HAN, self).__init__()
        self.use_gpu = use_gpu
        self.word_emb_dim = word_emb_dim
        self.emb_dim = emb_dim
        # Review Embedding
        self.review_lookup = nn.Embedding(review_vocab_size, word_emb_dim, padding_idx=1)
        self.review_sent_emb = AttendedSeqEmbedding(input_dim=word_emb_dim, hidden_dim=rnn_hidden_dim,
                                                    output_dim=emb_dim, rnn_type=rnn_type)
        self.review_emb = AttendedSeqEmbedding(input_dim=emb_dim, hidden_dim=rnn_hidden_dim,
                                               output_dim=emb_dim, rnn_type=rnn_type)
        self.empty_review_emb = nn.Parameter(torch.Tensor(emb_dim))
        self.empty_review_emb.data.normal_(0, 0.1)

        # Summary Embedding if required and classifier
        self.use_summary = use_summary
        self.combined_lookup = combined_lookup
        self.rnn_type = rnn_type
        self.use_summ_mlp = use_summ_mlp
        if use_summary:
            if not combined_lookup:
                self.summary_lookup = nn.Embedding(summary_vocab_size, word_emb_dim, padding_idx=1)
            else:
                self.summary_lookup = self.review_lookup
            if self.rnn_type == 'lstm':
                self.summary_rnn = nn.LSTM(input_size=word_emb_dim, hidden_size=rnn_hidden_dim, bidirectional=True, batch_first=True)
            else:
                self.summary_rnn = nn.GRU(input_size=word_emb_dim, hidden_size=rnn_hidden_dim, bidirectional=True, batch_first=True)
            rnn_output_dim = 2 * rnn_hidden_dim
            if self.use_summ_mlp:
                self.summary_mlp = nn.Sequential(
                    nn.Linear(rnn_output_dim, emb_dim),
                    nn.Tanh())
            self.empty_summary_emb = nn.Parameter(torch.Tensor(emb_dim))
            self.empty_summary_emb.data.normal_(0, 0.1)
            self.classifier = nn.Linear(emb_dim * 2, output_dim)
        else:
            self.classifier = nn.Linear(emb_dim, output_dim)

    def embed(self, sequences, seqlens, seq_emb, empty_seq_emb):
        nonempty = (seqlens != 0).nonzero()
        nonempty_seq_embs = seq_emb(sequences, seqlens[nonempty])
        if np.count_nonzero(seqlens != 0) < len(seqlens):
            seq_embs = Variable(torch.Tensor(len(seqlens), self.emb_dim))
            if self.use_gpu:
                seq_embs = seq_embs.cuda()
            # Pdb().set_trace()
            seq_embs[nonempty] = nonempty_seq_embs
            empty = (seqlens == 0).nonzero()
            seq_embs[empty] = empty_seq_emb.expand(len(empty[0]), self.emb_dim)
        else:
            seq_embs = nonempty_seq_embs
        return seq_embs

    def summary_emb(self, summaries, summlens):
        packed_input, indices = pack(summaries, summlens, self.use_gpu)
        # self.summary_rnn.flatten_parameters()
        _, h = self.summary_rnn(packed_input)
        if self.rnn_type == 'lstm':
            h = h[0]
        summ_embs = torch.cat([h[0], h[1]], dim=1)
        orig_indices = [0] * indices.shape[0]
        for i in range(indices.shape[0]):
            orig_indices[indices[i]] = i
        summ_embs = summ_embs[torch.LongTensor(orig_indices).cuda()]
        return summ_embs

    def compute_summary_embs(self, summaries):
        words = Variable(torch.LongTensor([word for summary in summaries for word in summary]))
        if self.use_gpu:
            words = words.cuda()
        summ_word_embs = self.summary_lookup(words)
        summlens = np.array([len(summary) for summary in summaries])
        summary_embs = self.embed(summ_word_embs, summlens, self.summary_emb, self.empty_summary_emb)
        if self.use_summ_mlp:
            summary_embs = self.summary_mlp(summary_embs)
        return summary_embs

    def compute_review_embs(self, reviews):
        words = Variable(torch.LongTensor([word for review in reviews for sent in review for word in sent]))
        if self.use_gpu:
            words = words.cuda()
        review_word_embs = self.review_lookup(words)

        # sentlens is a list of list of #words in each sentence in each review
        sentlens = np.array([len(sent) for review in reviews for sent in review])
        review_sent_embs = self.review_sent_emb(review_word_embs, sentlens)

        # reviewlens is a list of #sentences in each review
        reviewlens = np.array([len(review) for review in reviews])
        review_embs = self.embed(review_sent_embs, reviewlens, self.review_emb, self.empty_review_emb)
        return review_embs

    # each review is a list of list of words
    def forward(self, reviews, summaries):
        try:
            review_embs = self.compute_review_embs(reviews)
            if self.use_summary:
                summary_embs = self.compute_summary_embs(summaries)
                embeddings = torch.cat([review_embs, summary_embs], dim=1)
            else:
                embeddings = review_embs
            outputs = self.classifier(embeddings)
        except Exception as e:
            print(e)
            Pdb().set_trace()
        return F.softmax(outputs, dim=1)
