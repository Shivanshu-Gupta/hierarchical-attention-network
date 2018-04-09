import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description='Preprocess audio reviews dataset using Spacy')
parser.add_argument('--input_file', default='', required=True, metavar='PATH')
parser.add_argument('--output_file', default='', required=True, metavar='PATH')
args = parser.parse_args()


def load_data(datafile):
    import html
    samples = [json.loads(line) for line in open(datafile).readlines()]
    data = {}
    data['review'] = [html.unescape(sample['reviewText']) for sample in samples]
    data['summary'] = [html.unescape(sample['summary']) for sample in samples]
    data['rating'] = np.array([sample['overall'] for sample in samples])

    classes = np.array([0, 1, 2])

    def target(rating):
        if rating <= 2:
            return classes[0]
        elif rating == 3:
            return classes[1]
        else:
            return classes[2]
    data['target'] = np.array([target(rating) for rating in data['rating']])

    return data


def dump_dataset(raw_data, outfile, summary=True, tag=False, lemma=False):
    import spacy
    with open(outfile, 'w') as outf:
        nlp = spacy.load('en_core_web_sm')
        review_docs = nlp.pipe(raw_data['review'])
        summ_docs = nlp.pipe(raw_data['summary'])
        for i, (review, summ, target) in enumerate(zip(review_docs, summ_docs, raw_data['target'])):
            sample = {}
            # REVIEW
            # remove stop-words and whitespace tokens
            review_valid = [[tok for tok in sent if not tok.is_stop and tok.text.strip() != ''] for sent in review.sents]
            # remove empty sentences
            review_valid = [sent for sent in review_valid if not len(sent) == 0]
            sample['review'] = [[tok.text.lower() for tok in sent] for sent in review_valid]
            if tag:
                # sample['review_tag'] = [[tok.tag for tok in sent] for sent in review_valid]
                sample['review_tag_'] = [[tok.tag_.lower() for tok in sent] for sent in review_valid]
            if lemma:
                sample['review_lemma'] = [[tok.lemma_.lower() for tok in sent] for sent in review_valid]

            # SUMMARY
            # remove stop-words and whitespace tokens
            summary_valid = [tok for tok in summ if not tok.is_stop and tok.text.strip() != '']
            sample['summary'] = [tok.text.lower() for tok in summary_valid]
            if tag:
                # sample['summary_tag'] = [tok.tag for tok in summary_valid]
                sample['summary_tag_'] = [tok.tag_.lower() for tok in summary_valid]
            if lemma:
                sample['summary_lemma'] = [tok.lemma_.lower() for tok in summary_valid]
            sample['target'] = int(target)
            outf.write(json.dumps(sample) + '\n')
            if i % 1000 == 0:
                print(i)


if __name__ == '__main__':
    print("Loading raw data from {}".format(args.input_file))
    data = load_data(args.input_file)
    print("Preprocessing data and writing to {}".format(args.output_file))
    dump_dataset(data, args.output_file)
