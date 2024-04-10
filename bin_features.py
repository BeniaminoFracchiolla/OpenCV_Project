# -*- coding: utf-8 -*-
"""
Compute the features of the object in the image


@author: eusebio
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np


def area(img_bin):
    '''
    Compute the AREA of the blob. If the image contains more than one blob, 
    the functions returns the overall area of the blobs in the image


    Parameters
    ----------
    img_bin : TYPE uint8 binary image
        DESCRIPTION. Binary image. Object with values != 0. 
        The image should have only one object

    Returns
    -------
    area : TYPE int  
        DESCRIPTION.  Number of active pixles in the image.

    '''
    rows, colums = img_bin.shape
    num_active_pixels = 0
    for i in range(rows):
        for j in range(colums):
            if img_bin[i, j] != 0:
                num_active_pixels += 1
    return num_active_pixels


def centroid(img_bin):
    '''
    Compute the centroid of the blob. If the image contains more than one blob, 
    the functions returns the centroid of the ensemble


    Parameters
    ----------
    img_bin : TYPE uint8 binary image
        DESCRIPTION. Binary image. Object with values != 0. 
        The image should have only one object

    Returns
    -------
    centroid : TYPE int tuple (c_i, c_j) 
        c_i: rows, y axis TYPE int
        c_j: columns, x axis
        DESCRIPTION. Center of gravity of the object or ensemble 

    '''
    rows, colums = img_bin.shape
    area = 0
    sum_i = 0
    sum_j = 0
    for i in range(rows):
        for j in range(colums):
            if img_bin[i, j]:
                sum_i += i
                sum_j += j
                area += 1

    c_i = round(sum_i/area)
    c_j = round(sum_j/area)
    centroid = (c_i, c_j)
    return centroid


def circularity(img_bin):
    ##cv2.floodFill(im_floodfill, mask, (0,0), 255);
    img_fill = np.zeros_like(img_bin)
    cnt, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_fill, [cnt[0]], 0, 1, -1)  # -1 solid
    area_filled = np.sum(img_fill)
    perimeter = cv2.arcLength(cnt[0], True)# true because the contour is closed
    circularity = 4 * np.pi * area_filled / (perimeter**2)

    # plt.imshow(img_fill*255, 'gray')
    # plt.title('Binary Image')
    # plt.axis('off')
    # plt.show()

    return circularity


def count_holes(img_bin):
    '''
    Count the number of holes in the image from a binary image (object in white)
    If the img contains only one blob will be the number of holes in the object
    Parameters
    ----------
    img_bin : TYPE bin imag uint8
        DESCRIPTION. Image with objects in white, holes in black

    Returns
    -------
    num_holes : TYPE int
        DESCRIPTION.

    '''
    # if we invert the binary image then the holes are the objects
    img_inverted = np.zeros_like(img_bin, np.uint8)
    img_inverted[img_bin==0]=255
    # labeling the holes
    num_labels, img_label = cv2.connectedComponents(img_inverted)
    num_holes = num_labels - 2  # opencv labels also the background
    return num_holes


def count_holes_from_hierarchy(hierarchy):
    '''
    Count the number of holes in the image from hierarchy
    If the img contains only one blob will be the number of holes in the object
    Parameters
    ----------
    hierarchy : TYPE int array N x 4
        DESCRIPTION. hierachy of contours 
        see https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
        findContours provides the hierarchy
        contours, hierarchy = cv2.findContours(
                                img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    Returns
    -------
    TYPE int
        DESCRIPTION. Number of holes in the blob

    '''
    num_holes = 0
    if hierarchy[0, 0, 2] < 0:  # there are no holes
        return 0
    else:  # There are holes. Let's count them
        hole = hierarchy[0, 0, 2]
        while hole > 0:
            num_holes += 1
            hole = hierarchy[0, hole, 0]
    return num_holes


def size_filter(img_labels, blob_sizes, min_size, max_size=-1):
    '''
    Filter the blobs in img_labels by size. 
    Only blobs with min_size <= area <= max_size will remain 

    Parameters
    ----------
    img_labels : TYPE
        DESCRIPTION. label image
    blob_sizes : TYPE np.array
        DESCRIPTION. array with the area of the blobs
    min_size : TYPE
        DESCRIPTION.
    max_size : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    num_blobs_ok : TYPE np.array
        DESCRIPTION. zero in the position of the rejected blobs
    img_blobs_ok : TYPE 
        DESCRIPTION. binary image with only the correct blobs

  '''
    # si no se especifica max_size, no hay que quitar blobs grandes
    if min_size != 0:
        blob_sizes[blob_sizes < min_size] = 0
    if max_size != -1:
        blob_sizes[blob_sizes > max_size] = 0
    # Ponemos a cero las etiquetas que no cumplan
    img_blobs_ok = np.zeros_like(img_labels)
    num_blobs_ok = 0
    for i in range(len(blob_sizes)):
        if blob_sizes[i] != 0:
            img_blobs_ok[img_labels == i] = 255
            num_blobs_ok += 1

    return num_blobs_ok, img_blobs_ok


def colour_labels(num_labels, labels):
    '''
    Display the labels with random colors

    Parameters
    ----------
    num_labels : TYPE int
        DESCRIPTION. Number of labels in image (including background, label 0)
    labels : TYPE gray level image with the labels
        DESCRIPTION. Image of labels. Background label 0

    Returns
    -------
    labeled_img : TYPE colour image
        DESCRIPTION. Output COLOR image, labels in random colors, background 0

    '''
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


if __name__ == '__main__':
    img_title = '../images/brida.png'
    img_title = '../images/circle.png'
    # Read the image in a greyscale mode
    img = cv2.imread(img_title, 0)

    # Optimal threshold value is determined automatically.
    # THRESH_BINARY_INV because objects must be white
    _, img_bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_white_pixels = area(img_bin)
    print('area =', num_white_pixels)

    c_i, c_j = centroid(img_bin)
    img_color = cv2.merge([img_bin, img_bin, img_bin])
    # draw a circle in computed centroid
    # centre,radius,color, width
    cv2.circle(img_color, (c_j, c_i), 5, (255, 0, 255), 2)

    number_of_holes = count_holes(img_bin)
    print('number of holes =', number_of_holes)

    circul = circularity(img_bin)
    print('circularity =', circul)

    plt.imshow(img_color, 'gray')
    plt.title('Binary Image')
    plt.axis('off')
    plt.show()
