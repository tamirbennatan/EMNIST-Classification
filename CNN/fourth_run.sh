#!/bin/bash

#######
# Fourth run on AWS
# This time, will use new and improved preprocessed data. 
# This time, use the circle area preprocessing scheme.
####

echo "4 Convolutions (first variant): in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 8 --rot 10 --X_path ~/data/rot/X_train.npy --y_path ~/data/rot/y_train.npy
echo "Done"

echo "6 Convolutions: in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 8 --rot 10 --X_path ~/data/rot/X_train.npy --y_path ~/data/rot/y_train.npy
echo "Done"
