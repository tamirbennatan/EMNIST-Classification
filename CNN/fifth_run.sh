#!/bin/bash

#######
# Fourth run on AWS
# This time, will use new and improved preprocessed data. 
# This time, use the properties of the smallest rotated rectangle. 
####



echo "4 Convolutions (first variant), batch of 16, circle: in training..."
python3 cnn.py --arch 1 -a --lr .00005 --batch 16 --rot 20 --X_path ~/data/circle/X_trainnorm.npy --y_path ~/data/circle/y_trainnorm.npy --subscript circle_nrom1
echo "Done"

echo "6 Convolutionsm batch of 16: in training..."
python3 cnn.py --arch 3 -a --lr .00005 --batch 16 --rot 20 --X_path ~/data/circle/X_trainnorm.npy --y_path ~/data/circle/y_trainnorm.npy --subscript circle_nrom1
echo "Done"

echo "4 Convolutions (first variant), batch of 16, rotation: in training..."
python3 cnn.py --arch 1 -a --lr .00005 --batch 16 --rot 20 --X_path ~/data/maxdim/X_trainnorm.npy --y_path ~/data/maxdim/y_trainnorm.npy --subscript rot_norm1
echo "Done"

echo "6 Convolutionsm batch of 16: in training..."
python3 cnn.py --arch 3 -a --lr .00005 --batch 16 --rot 20 --X_path ~/data/maxdim/X_trainnorm.npy --y_path ~/data/maxdim/y_trainnorm.npy --subscript rot_norm1
echo "Done"
