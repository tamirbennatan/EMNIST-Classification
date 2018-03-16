#!/bin/bash

#######
# Second run on AWS. 
# This time, decrease the learning rate, increase the batch sizes. 
# This will lead to slower training. 
####

echo "4 Convolutions (first variant): in training..."
python3 cnn.py --arch 1 -a --lr .00001 --batch 16
echo "Done."
echo

echo "6 Convolutions : in training..."
python3 cnn.py --arch 3 -a --lr .00001 --batch 16
echo "Done."
echo