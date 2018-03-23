#!/bin/bash

INPUT_FILE=$1
INPUT_FILE_CLEANED=$INPUT_FILE.clean
INPUT_FILE_PREPROCESSED=$INPUT_FILE.preprocessed

echo "Cleaning raw data..."
./clean.sh $INPUT_FILE

echo "Preprocessing cleaned data..."
python preprocess.py --input_file $INPUT_FILE_CLEANED --output_file $INPUT_FILE_PREPROCESSED

# For training data:
# echo "Building vocab for training data..."
# python vocab.py --train_data $INPUT_FILE_PREPROCESSED --review_vocab review_vocab.pkl --summary_vocab summary_vocab.pkl
# python vocab.py --train_data $INPUT_FILE_PREPROCESSED --vocab_file comb_vocab_pruned-10000.pkl

python main.py --config config_test.yaml --testfile $INPUT_FILE_PREPROCESSED --outputfile $2
