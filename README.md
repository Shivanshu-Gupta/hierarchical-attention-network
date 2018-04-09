# Hierarchical Attention Network for Sentiment Classification
A PyTorch implementation of the [Hierarchical Attention Network] for Sentiment Analysis 
on the [Amazon Product Reviews] datasets. The system uses the review text and the summary
text to classify the reviews as one of positive, negative or neutral. These classes 
correspond to ratings 4-5, 1-2 and 3 respectively in the dataset.

## Requirements
- python 3.5
- [pytorch]
- [torchtext]
- [spacy]

## Organisation
The code in the repository are organised in following modules:
- **main.py**: driver code
- **model.py**: Hierachical Attention Network implementation
- **train.py**: training/validation/testing code
- **preprocess.py**: data preprocessing code
- **vocab.py**: code for building vocab
- **dataset.py**: custom pytorch dataset for review data
- **utils.py**: logging, config generation, experiment analysis scripts

Following utility scripts have been added for training/testing:
- **train.sh**: will clean and preprocess train data, generate vocabulary pickles, 
and then train the model on the preprocessed data.
- **test.sh**: clean and preprocess test data, evaluate model on the preprocessed 
data and write model predictions to file.

## Usage
```sh
$ ./train.sh <train_data_json>
$ ./test.sh <test_data_json> <result_file>
```

## References
- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) (Yang et al, 2016)
- [Embed, Encode, Attend, Predict](https://explosion.ai/blog/deep-learning-formula-nlp)
- http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/ 
- http://anie.me/On-Torchtext/

[Hierarchical Attention Network]: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
[Amazon Product Reviews]: http://jmcauley.ucsd.edu/data/amazon/
[pytorch]: http://pytorch.org/
[torchtext]: https://github.com/pytorch/text
[spacy]: https://spacy.io/
