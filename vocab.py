import json
import argparse
import pickle
from torchtext.data import Field

parser = argparse.ArgumentParser(description='Build review and summary vocabulary from preprocessed data using torchtext')
parser.add_argument('--train_data', default='', required=True, metavar='PATH')
parser.add_argument('--vocab_file', default='', required=True, metavar='PATH')
args = parser.parse_args()

if __name__ == '__main__':
    train_data_file = args.train_data
    vocab_file = args.vocab_file
    print("Loading train data from {}".format(train_data_file))
    data = [json.loads(line) for line in open(train_data_file).readlines()]
    print("Building vocab...")
    reviews = [[word for sent in sample['review'] for word in sent] + sample['summary'] * 2 for sample in data]
    review_field = Field()
    review_field.build_vocab(reviews, min_freq=2, vectors="glove.6B.200d")
    print("Dumping vocab to {}".format(vocab_file))
    pickle.dump(review_field.vocab, open(vocab_file, 'wb'))
