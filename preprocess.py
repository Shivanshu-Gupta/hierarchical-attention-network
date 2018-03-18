import os
import argparse
import json
import spacy
import pickle
import html
import numpy as np

nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description='SVM-based Sentiment Analyzer')
parser.add_argument('--input_file', default='', required=True, metavar='PATH')
parser.add_argument('--output_file', default='', required=True, metavar='PATH')
args = parser.parse_args()

def load_data(datafile):
    samples = [json.loads(line) for line in open(datafile).readlines()]
    data = {}
    data['review'] = [html.unescape(sample['reviewText']) for sample in samples]
    data['summary'] = [html.unescape(sample['summary']) for sample in samples]
    data['rating'] = np.array([sample['overall'] for sample in samples])

    classes = np.array([-1, 0, 1])

    def target(rating):
        if rating <= 2:
            return classes[0]
        elif rating == 3:
            return classes[1]
        else:
            return classes[2]
    data['target'] = np.array([target(rating) for rating in data['rating']])

    return data


# def preprocess_old(data, outfile):
#     with open(outfile, 'w') as outf:
#         review_docs = nlp.pipe(data['review'])
#         summ_docs = nlp.pipe(data['summary'])
#         for i, (review, summ, target) in enumerate(zip(review_docs, summ_docs, data['target'])):
#             sample = {}
#             sample['review'] = [[(tok.text, tok.pos_, tok.lemma_) for tok in sent if not tok.is_stop and tok.text.strip() != ''] for sent in review.sents]
#             sample['summary'] = [(tok.text, tok.pos_, tok.lemma_) for tok in summ if not tok.is_stop and tok.text.strip() != '']
#             sample['target'] = int(target)
#             outf.write(json.dumps(sample) + '\n')
#             if i % 1000 == 0:
#                 print(i)
# 
# def preprocessed_json_to_dataset(datafile, outfile):
#     with open(outfile, 'w') as outf:
#         for i, line in enumerate(open(datafile)):
#             sample = json.loads(line)
#             sample['review_pos_'] = [[tok[1] for tok in sent] for sent in sample['review']]
#             sample['review_lemma'] = [[tok[2] for tok in sent] for sent in sample['review']]
#             sample['review'] = [[tok[0] for tok in sent] for sent in sample['review']]
#             sample['summary_pos_'] = [tok[1] for tok in sample['summary']]
#             sample['summary_lemma'] = [tok[2] for tok in sample['summary']]
#             sample['summary'] = [tok[0] for tok in sample['summary']]
#             outf.write(json.dumps(sample) + '\n')
#             if i % 1000 == 0:
#                 print(i)

def preprocess(data, outfile):
    with open(outfile, 'w') as outf:
        review_docs = nlp.pipe(data['review'])
        summ_docs = nlp.pipe(data['summary'])
        for i, (review, summ, target) in enumerate(zip(review_docs, summ_docs, data['target'])):
            sample = {}
            review_valid = [[tok for tok in sent if not tok.is_stop and tok.text.strip() != ''] for sent in review.sents]
            sample['review'] = [[tok.text for tok in sent] for sent in review_valid]
            sample['review_pos'] = [[tok.pos for tok in sent] for sent in review_valid]
            sample['review_pos_'] = [[tok.pos for tok in sent] for sent in review_valid]
            sample['review_lemma'] = [[tok.lemma_ for tok in sent] for sent in review_valid]
            summary_valid = [tok for tok in summ if not tok.is_stop and tok.text.strip() != '']
            sample['summary'] = [tok.text for tok in summary_valid]
            sample['summary_pos'] = [tok.pos for tok in summary_valid]
            sample['summary_pos_'] = [tok.pos_ for tok in summary_valid]
            sample['summary_lemma'] = [tok.lemma_ for tok in summary_valid]
            sample['target'] = int(target)
            outf.write(json.dumps(sample) + '\n')
            if i % 1000 == 0:
                print(i)

if __name__ == '__main__':
    data = load_data(args.input_file)
    preprocess(data, args.output_file)
