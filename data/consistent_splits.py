"""
This script is to ensure that we have consistent training/validation splits. 
Make sure that you have the following viles saved into the /raw folder:

- train_x.csv
- train_y.csv
- test_x.csv

This script will split the training data into an 80/20 training/validation split. 
The resulting data will be saved as numpy arrays (.npy) in the `/pre_split` folder.
"""