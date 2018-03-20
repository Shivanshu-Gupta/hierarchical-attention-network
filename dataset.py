import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from IPython.core.debugger import Pdb


class ReviewsDataset(Dataset):
    def __init__(self, data, review_vocab, summary_vocab=None):
        if isinstance(data, str):
            datafile = data
            data = [json.loads(line) for line in open(datafile).readlines()]
        self.reviews = [sample['review'] for sample in data]
        self.summaries = [sample['summary'] for sample in data]
        self.targets = [sample['target'] for sample in data]
        self.review_vocab = review_vocab
        self.summary_vocab = summary_vocab

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        stoi = self.review_vocab.stoi
        review = self.reviews[idx]
        target = self.targets[idx]
        numerical = [[stoi[w] for w in sent] for sent in review if len(sent) > 0]
        return numerical, target

    def get_sampler(self):
        class_sample_count = np.unique(self.targets, return_counts=True)[1]
        weight = 1. / class_sample_count
        sample_weights = torch.from_numpy(weight[self.targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler
