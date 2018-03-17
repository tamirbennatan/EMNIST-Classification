#!/bin/bash

#######
# Second run on AWS. 
# This time, use the circle area preprocessing scheme.
####

echo "4 Convolutions (first variant): in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 16 --preprocess circle --rot 15
echo "Done."
echo

echo "6 Convolutions : in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 16 --preprocess circle --rot 15
echo "Done."
echo