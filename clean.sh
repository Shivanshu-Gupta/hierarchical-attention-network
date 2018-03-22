#!/bin/bash

INPUT_FILE=$1
tr -cd '\11\12\15\40-\176' < $INPUT_FILE > $INPUT_FILE.clean
sed -i 's/\([.*!?]\)\1\{3,\}/\1\1\1 /g' $INPUT_FILE.clean
sed -i 's/\([A-Za-z]\)\1\{3,\}/\1\1\1 /g' $INPUT_FILE.clean
