"""
A script to apply simple preprocessing to the Data. 

Requirements:
- Run this script from within the /preprocess directory. You should execute the
  command `python3 preprocess_simple.py`. 
- All the data files `x_train.csv, x_test.csv, y_train.csv` must be in the 
  /data/raw directory. 
- Run this with Python 3 (cv2 functionality requires python3)

The preprocessing done are:
- Black out the backgrounds of each picuture
- Extract the number which occupies the largest rectangle from each image
- Rescale to 28x28 image
- Write to the /data/preprocessed/basic/ directory. 

This script will write numpy binary files. Later, to read them, do for example:
```
with open([path to `data/preprocessed/basicy_train.npy`], "rb") as handle:
	y_train = np.load(handle)
```
"""
import numpy as np
import pandas as pd
import copy
from PIL import Image, ImageOps
from skimage.transform import resize
import cv2
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--circle", action="store_true",
                       help="Use circle min area to extract smallest picutre")
argparser.add_argument("--subscript", dest = "subscript", type = str, default = "",
				 help = "Subscript onto file names")

args = argparser.parse_args()
circle = args.circle

if circle:
	preprocess = "circle"
else:
	preprocess = "basic"

subscript = args.subscript


def to_black_or_white(original):
    """
    Input: Numpy array of dimension [numimages]X[width]X[Height]. 
    Pixel intensities are assumed to be 8 bit integers (0 - 255).
    
    Return: Numpy array of the same dimensions, 
    though all pixels that are not white (255) become black. 
    """
    # copy, so that we don't destroy original dat by reference. 
    images = copy.deepcopy(original)
    # are the pixels not white? 
    notwhite = images < 255
    # if its not white, convet to black
    images[notwhite] = 0
    return images.astype("uint8")


def getcontours(image):
    """
    Return a list of contours for an image
    """
    # find all the contours in that image
    im2, contours, hierarchy = cv2.findContours(image, 
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # return only the contours of greater than one pixel in size
    contours = [cnt for cnt in contours if len(cnt) > 1]
    return contours

def paste_onto_black(cropped, desiredsize = 28):
    
    # get the ratio with which we'll scale the size
    ratio = desiredsize/float(max(cropped.shape))
    # the scaled size
    newsize = tuple([int(ratio*s) for s in cropped.shape])
    # resize the image, while keeping dimensions
    im = resize(cropped, newsize, preserve_range=True, mode="constant")
    
    # difference in width/height
    dim_diff = im.shape[0] - im.shape[1]
    # higher than it's wide
    if dim_diff > 0:
        # find the number of colums you need to add to the left/right
        lcols = dim_diff //2
        rcols = dim_diff - lcols
        # template to add to both sides
        temp = np.repeat(0, im.shape[0])
        temp = temp.reshape((temp.shape[0], -1))
        # Add the neccessary number of columns on left
        for i in range(lcols):
            im = np.hstack((temp, im))
        # Add the neccessary number of columns on the right
        for i in range(rcols):
            im = np.hstack((im, temp))
    # wider than it's wide
    elif dim_diff < 0:
        # find the number of rows you need to add to the top/button
        trows = abs(dim_diff) //2 
        brows = abs(dim_diff) - trows
        # template to add to both sides
        temp = np.repeat(0, im.shape[1])
        # Add neccessary rows to the top
        for i in range(trows):
            im = np.vstack((temp, im))
        # Add neccessary rows to the bottom
        for i in range(brows):
            im = np.vstack((im, temp))
    # image is square - do no padding neccessary
    else:
        pass
    return im

"""
From a list countours, return the largest enclosing rectangle. 
Definition of "largest" is the largest enclosing rectangle if `circle` is False,
otherwise its the digit with the largest enclosing circle. 
"""
def largest_enclosingRect(img, contours, circle = False):
    # keep track of largest contour/area seen so far
    largest_cnt, max_area = None, -1
    
    if not circle:
        for cnt in contours:
            # top left corner, and width/height of the image
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            # is this the largest image we've seen so far? 
            if area > max_area:
                max_area = area
                largest_cnt = cnt
    else:
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            # The area is proportinal to the radius - don't need to actually compute pi*r^2
            if radius > max_area:
                max_area = radius
                largest_cnt = cnt
    
    # Get the enclosing rectangle of the largest contour
    x,y,w,h = cv2.boundingRect(largest_cnt)
    # crop the image, and return
    cropped=img[y:y+h,x:x+w]
    return(cropped) 

def crop_images(images, desiredsize = 28, circle = False):
    """
    Given an array of 64x64 black/white images, 
    return an array of the same length, where the contour with the largest _enclosing rectangle_ 
    is extracted from each image. 
    
    Each number is scaled to size (desiredsize,desiredsize).
    """
    # accumulate a modified dataset
    modified = list()
    for im in images:
        # find all the contours in that image
        contours = getcontours(im)
        # get the largest enclosing rectangle
        enclosing_rect = largest_enclosingRect(im, contours, circle = circle)
        # paste it onto a black canvas
        scaled = paste_onto_black(enclosing_rect)
        # put it in our accumulated (modified) dataset. 
        modified.append(scaled)
    # when appending, images are unrolled. Roll them back up. 
    modified = np.array(modified)
    return(modified)

def normalize(arr):
    arr = copy.copy(arr)
    blackpix = arr == 0
    whitepix = np.logical_not(blackpix)
    arr[blackpix] = 1
    arr[whitepix] = 0
    return arr

def main():
    """
    Load the training and test sets. 
    Also load the labels, so that you can save them more nicely. 
    """
    print("Loading Data...")
    # X_test = np.loadtxt("../data/raw/test_x.csv", delimiter=",") # load from text 
    # X_train = np.loadtxt("../data/raw/train_x.csv", delimiter=",")
    # y_train = np.loadtxt("../data/raw/train_y.csv", delimiter=",") 
    # X_train = X_train.reshape(-1, 64, 64) 
    # X_test = X_test.reshape(-1, 64, 64) # reshape 
    # y_train = y_train.reshape(-1, 1) 
    X_test = np.load("../data/raw/X_test.npy")
    X_train = np.load("../data/raw/X_train.npy")
    y_train = np.load("../data/raw/y_train.npy")
    print("Done.")
    print("Done.")

    """
    Convert both the training and test sets to black/white images. 
    """
    print("Converting to black/white...")
    X_train = to_black_or_white(X_train)
    X_test = to_black_or_white(X_test)
    print("Done.")

    """
    Crop the biggest numbers from each image in the training/test sets:
    """
    print("Cropping largest images from training data...")
    X_train_cropped = crop_images(X_train, circle = circle)
    print("Done.")
    print("Cropping largest images from test data...")
    X_test_cropped = crop_images(X_test, circle = circle)
    print("Done.")

    """
    Normalize values to range {0,1}
    """
    print ("Normalizing....")
    X_train_cropped = normalize(X_train_cropped)
    X_test_cropped = normalize(X_test_cropped)
    print("Done.")

    """
    Save the numpy arrays 
    """
    # print("Saving files...")
    # if not os.path.exists("../data/preproccessed/basic"):
    #     os.mkdir(os.path.relpath("../data/preproccessed/basic"), exist_ok=True)

    # if not os.path.exists("../data/preproccessed/circle"):
    #     os.mkdir(os.path.relpath("../data/preproccessed/circle"), exist_ok=True)

    # Save the labels
    with open("../data/preproccessed/%s/y_train%s.npy" % (preprocess, subscript), "wb") as handle:
        np.save(handle,y_train)
    # save the training data
    with open("../data/preproccessed/%s/X_train%s.npy" % (preprocess, subscript), "wb") as handle:
        np.save(handle,X_train_cropped)
    # save the test data
    with open("../data/preproccessed/%s/X_test%s.npy" % (preprocess, subscript), "wb") as handle:
        np.save(handle,X_test_cropped)

    print("Done. :D")


if __name__ == '__main__':
    main()
