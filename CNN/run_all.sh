#!/bin/bash

#######
# A simple script to run all the different archetectures defined. 
# Designed to be run on AWS servers, not locally.
####

echo "3 Convolutions: in training..."
python3 cnn.py --arch 0 -a
echo "Done."
echo

echo "4 Convolutions (first variant): in training..."
python3 cnn.py --arch 1 -a
echo "Done."
echo

echo "4 Convolutions (second variant): in training..."
python3 cnn.py --arch 2 -a
echo "Done."
echo

echo "6 Convolutions : in training..."
python3 cnn.py --arch 3 -a
echo "Done."
echo