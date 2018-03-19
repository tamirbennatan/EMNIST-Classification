#!/bin/bash

#######
# Fourth run on AWS
# This time, will use new and improved preprocessed data. 
# This time, use the properties of the smallest rotated rectangle. 
####

echo "4 Convolutions (first variant), no generation: in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 1 --rot 0 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"

echo "6 Convolutions, no generation: in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 1 --rot 0 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"

echo "4 Convolutions (first variant), batch of 8: in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 8 --rot 10 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"

echo "6 Convolutionsm batch of 8: in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 8 --rot 10 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"


echo "4 Convolutions (first variant), batch of 16: in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 16 --rot 20 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"

echo "6 Convolutionsm batch of 16: in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 16 --rot 20 --X_path ~/data/maxdim/X_train.npy --y_path ~/data/maxdim/y_train.npy
echo "Done"
