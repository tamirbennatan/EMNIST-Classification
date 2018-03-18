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
argparser.add_argument("-a", "--area", action="store_true",
                       help="Use the rotated rectangle area to define 'largest'\n" +\
                       "Otherwise use the largest dimension of the rotated rectangle.")

args = argparser.parse_args()
area = args.area

if area:
    metric = "maxarea"
else:
    metric = "maxdim"

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

def crop_enclosingRect(img, cnt, padding = 2):
    # get the dimension of the image (assumed to be square)
    size = img.shape[0] - 1 # minus 1 because zero indexed 
    x,y,w,h = cv2.boundingRect(cnt)
    """
    Add `padding` pixels on the lef/right of the enclosing rectangle.
    Need to make sure, however, that you don't go over the end of the image's range. 
    """
    # update the x/y to add padding. 
    y0_new = max(0, y - padding)
    y1_new = min(size, y + h + padding)
    h_new = y1_new - y0_new
    
    x0_new = max(0, x - padding)
    x1_new = min(size,x + w + padding)
    w_new = x1_new - x0_new
    
    cropped=img[y0_new:y0_new+h_new,x0_new:x0_new+w_new]
    return(cropped)

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

def max_area_contour(contours):
    areas = [cv2.contourArea(cnt)for cnt in contours]
    return contours[np.argmax(areas)]


def rect_area(img,rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    
    xs, ys = tuple(zip(*pts))
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    return width*height

def rect_maxdim(img,rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    
    xs, ys = tuple(zip(*pts))
    maxdim = max(max(xs) - min(xs), max(ys) - min(ys))
    return maxdim

def largest_cnt_rot(img, contours, metric = "maxdim"):
    # get the rectangles for each contour
    rects = [cv2.minAreaRect(cnt) for cnt in contours]
    assert len(rects) == len(contours)
    # keep track of the largest contour you've seen, and metric
    largest_cnt, largest_metric = None, -1
    # the function to evaluate the "largeness" of a rectangle with
    metric_func = (rect_maxdim if metric == "maxdim" else rect_area)
    for i in range(len(contours)):
        m = metric_func(img, rects[i])
        if m > largest_metric:
            largest_metric = m
            largest_cnt = contours[i]
    return largest_cnt

def rot_crop(images, desiredsize = 28, metric = "maxdim"):
    """

    """
    # accumulate a modified dataset
    modified = np.ndarray((0, desiredsize, desiredsize), dtype = "uint8")
    
    for im in images:
        # find all the contours in that image
        contours = getcontours(im)
        # get the 'largest' contour, per inputed definition of largest
        largest_cnt = largest_cnt_rot(im, contours, metric = metric)
        # get the rectangle that encloses the "largest" contour
        enclosing_rect = crop_enclosingRect(im, largest_cnt)
        # paste it onto a black canvas
        scaled = paste_onto_black(enclosing_rect)
        # put it in our accumulated (modified) dataset. 
        modified = np.append(modified, np.array(scaled))
    # when appending, images are unrolled. Roll them back up. 
    modified = modified.reshape((images.shape[0],desiredsize,desiredsize))
    return(modified)

def main():
    """
    Load the training and test sets. 
    Also load the labels, so that you can save them more nicely. 
    """
    print("Loading Data...")
    X_test = np.loadtxt("../data/raw/test_x.csv", delimiter=",") # load from text 
    X_train = np.loadtxt("../data/raw/train_x.csv", delimiter=",")
    y_train = np.loadtxt("../data/raw/train_y.csv", delimiter=",") 
    X_train = X_train.reshape(-1, 64, 64) 
    X_test = X_test.reshape(-1, 64, 64) # reshape 
    y_train = y_train.reshape(-1, 1) 
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
    X_train_cropped = rot_crop(X_train, metric = metric)
    print("Done.")
    print("Cropping largest images from test data...")
    X_test_cropped = rot_crop(X_test, circle = circle)
    print("Done.")

    """
    Save the numpy arrays 
    """
    # print("Saving files...")
    # if not os.path.exists("../data/preproccessed/rot"):
       #     os.mkdir(os.path.relpath("../data/preproccessed/rot"), exist_ok=True)
    #    os.mkdir(os.path.relpath("../data/preproccessed/rot/%s") % metric, exist_ok=True)

    # Save the labels
    with open ("../data/preproccessed/rot/%s/y_train.npy" % metric, "wb") as handle:
        np.save(handle,y_train)
    # save the training data
    with open("../data/preproccessed/rot/%s/X_train.npy" % metric, "wb") as handle:
        np.save(handle,X_train_cropped)
    # save the test data
    with open("../data/preproccessed/rot/%s/X_test.npy" % metric, "wb") as handle:
        np.save(handle,X_test_cropped)

    print("Done. :D")


if __name__ == '__main__':
    main()
