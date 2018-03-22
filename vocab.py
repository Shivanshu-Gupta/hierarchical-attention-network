import json
import argparse
import pickle
from torchtext.data import Field

parser = argparse.ArgumentParser(description='Build review and summary vocabulary from preprocessed data using torchtext')
parser.add_argument('--train_data', default='', required=True, metavar='PATH')
parser.add_argument('--review_vocab', default='', metavar='PATH')
parser.add_argument('--summary_vocab', default='', metavar='PATH')
args = parser.parse_args()

# usage: build_dump_vocab('audio_train_dataset.json', 'review_vocab.pkl', 'summary_vocab.pkl')

if __name__ == '__main__':
    train_data_file = args.train_data
    review_vocab_file = args.review_vocab
    summary_vocab_file = args.summary_vocab
    if review_vocab_file == '' and summary_vocab_file == '':
        exit()
    print("Loading train data from {}".format(train_data_file))
    data = [json.loads(line) for line in open(train_data_file).readlines()]
    if review_vocab_file is not None:
        print("Building review vocab...")
        reviews = [[word for sent in sample['review'] for word in sent] for sample in data]
        review_field = Field()
        review_field.build_vocab(reviews, min_freq=5, vectors="glove.6B.200d")
        print("Dumping review vocab to {}".format(review_vocab_file))
        pickle.dump(review_field.vocab, open(review_vocab_file, 'wb'))
    if summary_vocab_file is not None:
        print("Building summary vocab...")
        summaries = [sample['summary'] for sample in data]
        summary_field = Field()
        summary_field.build_vocab(summaries, min_freq=1, vectors="glove.6B.200d")
        print("Dumping summary vocab to {}".format(summary_vocab_file))
        pickle.dump(summary_field.vocab, open(summary_vocab_file, 'wb'))
