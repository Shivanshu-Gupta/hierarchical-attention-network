# Hierarchical Attention Network for Sentiment Classification
A PyTorch implementation of the [Hierarchical Attention Network] for Sentiment Analysis 
on the [Amazon Product Reviews] datasets. The system uses the review text and the summary
text to classify the reviews as one of positive, negative or neutral. These classes 
correspond to ratings 4-5, 1-2 and 3 respectively in the dataset.

## Requirements
- python 3.5
- [pytorch]
- [torchtext]

## Organisation
The code in the repository are organised as follows:
- *model.py*: custom GRU
- *train.py*: training/validation/testing code
- *main.py*: driver code
- *dataset.py*: custom pytorch dataset for review data
- *preprocess.py*: data preprocessing code
- *vocab.py*: code for building vocab
- *utils.py*: logging, config generation, experiment analysis scripts

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

