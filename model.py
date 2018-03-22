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


class AttendedSeqEmbedding(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50, output_dim=100, bidirectional=True, use_gpu=True, batch_first=True):
        super(AttendedSeqEmbedding, self).__init__()
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
        self.context_vector.data.normal_(0, 0.1)

    def forward(self, sequences, seqlens):
        # print(seqlens)
        assert(sequences.size() == (seqlens.sum(), self.input_dim))
        padded_inputs = Variable(torch.zeros(len(seqlens), int(seqlens.max()), self.input_dim))
        if self.use_gpu:
            padded_inputs = padded_inputs.cuda()
        begin_idx = 0
        for idx, length in enumerate(seqlens):
            if length > 0:
                padded_inputs[idx, :length] = sequences[begin_idx:begin_idx + length]
            begin_idx += length
        # print(padded_inputs.shape)
        indices = np.argsort(-seqlens)
        # Pdb().set_trace()
        seqlens = seqlens[indices]
        padded_inputs = padded_inputs[torch.LongTensor(indices).cuda()]

        packed_input = pack_padded_sequence(padded_inputs, seqlens, batch_first=self.batch_first)
        # print(packed_input)
        self.gru.flatten_parameters()
        packed_output, _ = self.gru(packed_input)
        padded_outputs, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # TODO: Check if applying attention after splitting output is faster.
        padded_outputs = self.mlp(padded_outputs)

        # undo sort
        orig_indices = [0] * indices.shape[0]
        for i in range(indices.shape[0]):
            orig_indices[indices[i]] = i
        padded_outputs = padded_outputs[torch.LongTensor(orig_indices).cuda()]
        seqlens = seqlens[orig_indices]

        # apply attention
        outputs = [padded_outputs[i, :seqlens[i]] for i in range(padded_outputs.size(0))]
        attn_weights = [F.softmax(output.matmul(self.context_vector), dim=0) for output in outputs]
        attended_outputs = [attn_weight.matmul(output) for attn_weight, output in zip(attn_weights, outputs)]
        attended_outputs = torch.stack(attended_outputs)
        return attended_outputs


class HAN(nn.Module):
    def __init__(self, review_vocab_size, summary_vocab_size, word_emb_dim=100, gru_hidden_dim=50, emb_dim=100, output_dim=3, use_summary=True, batch_size=64, use_gpu=True):
        super(HAN, self).__init__()
        self.use_gpu = use_gpu
        self.word_emb_dim = word_emb_dim
        self.emb_dim = emb_dim
        # Review Embedding
        self.review_lookup = nn.Embedding(review_vocab_size, word_emb_dim, padding_idx=1)
        self.review_sent_emb = AttendedSeqEmbedding(input_dim=word_emb_dim, hidden_dim=gru_hidden_dim,
                                                    output_dim=emb_dim, bidirectional=True)
        self.review_emb = AttendedSeqEmbedding(input_dim=emb_dim, hidden_dim=gru_hidden_dim,
                                               output_dim=emb_dim, bidirectional=True)
        self.empty_review_emb = nn.Parameter(torch.Tensor(emb_dim))
        self.use_summary = use_summary
        if use_summary:
            # Summary Embedding
            self.summary_lookup = nn.Embedding(summary_vocab_size, word_emb_dim, padding_idx=1)
            self.summary_gru = nn.GRU(input_size=word_emb_dim, hidden_size=gru_hidden_dim, bidirectional=True, batch_first=True)
            self.empty_summary_emb = nn.Parameter(torch.Tensor(emb_dim))
            self.classifier = nn.Linear(emb_dim * 2, output_dim)
        else:
            self.classifier = nn.Linear(emb_dim, output_dim)

    def summary_emb(self, summary_word_embs, summary_lengths):
        padded_inputs = Variable(torch.zeros(len(summary_lengths), int(summary_lengths.max()), self.word_emb_dim))
        if self.use_gpu:
            padded_inputs = padded_inputs.cuda()
        begin_idx = 0
        for idx, length in enumerate(summary_lengths):
            if length > 0:
                padded_inputs[idx, :length] = summary_word_embs[begin_idx:begin_idx + length]
            begin_idx += length
        # print(padded_inputs.shape)
        indices = np.argsort(-summary_lengths)
        # print(indices)
        summary_lengths = summary_lengths[indices]
        padded_inputs = padded_inputs[torch.LongTensor(indices).cuda()]

        packed_input = pack_padded_sequence(padded_inputs, summary_lengths, batch_first=True)
        # print(packed_input)
        self.summary_gru.flatten_parameters()
        _, h = self.summary_gru(packed_input)
        summary_embs = torch.cat([h[0], h[1]], dim=1)
        orig_indices = [0] * indices.shape[0]
        for i in range(indices.shape[0]):
            orig_indices[indices[i]] = i
        summary_embs = summary_embs[torch.LongTensor(orig_indices).cuda()]
        summary_lengths = summary_lengths[orig_indices]
        return summary_embs

    def compute_summary_embs(self, summaries):
        words = Variable(torch.LongTensor([word for summary in summaries for word in summary]))
        if self.use_gpu:
            words = words.cuda()
        summary_word_embs = self.summary_lookup(words)
        summary_lengths = np.array([len(summary) for summary in summaries])
        nonempty_summaries = (summary_lengths != 0).nonzero()
        nonempty_summary_embs = self.summary_emb(summary_word_embs, summary_lengths[nonempty_summaries])
        if np.count_nonzero(summary_lengths != 0) < len(summary_lengths):
            summary_embs = Variable(torch.Tensor(len(summaries), self.emb_dim))
            if self.use_gpu:
                summary_embs = summary_embs.cuda()
            # Pdb().set_trace()
            summary_embs[nonempty_summaries] = nonempty_summary_embs
            empty_summaries = (summary_lengths == 0).nonzero()
            summary_embs[empty_summaries] = self.empty_summary_emb.expand(len(empty_summaries[0]), self.emb_dim)
        else:
            summary_embs = nonempty_summary_embs
        return summary_embs

    def compute_review_embs(self, reviews):
        words = Variable(torch.LongTensor([word for review in reviews for sent in review for word in sent]))
        if self.use_gpu:
            words = words.cuda()
        review_word_embs = self.review_lookup(words)

        # sent_lengths is a list of list of #words in each sentence in each review
        sent_lengths = np.array([len(sent) for review in reviews for sent in review])
        review_sent_embs = self.review_sent_emb(review_word_embs, sent_lengths)

        # review_lengths is a list of #sentences in each review
        review_lengths = np.array([len(review) for review in reviews])
        nonempty_reviews = (review_lengths != 0).nonzero()
        nonempty_review_embs = self.review_emb(review_sent_embs, review_lengths[nonempty_reviews])
        if np.count_nonzero(review_lengths != 0) < len(review_lengths):
            review_embs = Variable(torch.Tensor(len(reviews), self.emb_dim))
            if self.use_gpu:
                review_embs = review_embs.cuda()
            # Pdb().set_trace()
            review_embs[nonempty_reviews] = nonempty_review_embs
            empty_reviews = (review_lengths == 0).nonzero()
            review_embs[empty_reviews] = self.empty_review_emb.expand(len(empty_reviews[0]), self.emb_dim)
        else:
            review_embs = nonempty_review_embs
        return review_embs

    # a review is a list of list of words
    def forward(self, reviews, summaries):
        review_embs = self.compute_review_embs(reviews)
        # Pdb().set_trace()
        if self.use_summary:
            summary_embs = self.compute_summary_embs(summaries)
            embeddings = torch.cat([review_embs, summary_embs], dim=1)
        else:
            embeddings = review_embs
        outputs = self.classifier(embeddings)
        return F.softmax(outputs, dim=1)
