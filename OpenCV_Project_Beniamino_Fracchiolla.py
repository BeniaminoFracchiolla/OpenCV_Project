# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:53:08 2023

@author: Beniamino Fracchiolla
"""
# =============================================================================
# LIBRARY CALL
# =============================================================================

import time
start_time = time.time()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import bin_features as bf
import lms 
import ransac

# =============================================================================
# FUNCTION LIST
# =============================================================================

def area_selection(img_bin,area):
    '''
    This function return only the image of a label.
    It choose the label based on the area feature, with tollerance of 10%, of white objects.
    
    Parameters
    ----------
    img_bin : Image already binarized
    
    area : integer number of area of e binaryzed label

    Returns
    -------
    img_selected: Image of label whit that label

    '''
    row,colum=img.shape
    
    num_labels, img_labels,stats,centroid=cv2.connectedComponentsWithStats(img_bin) #only white objects
    selected_label=0

    # Finding white label on image based on area feature
    for i in range(num_labels):
        if 0.9*area < stats[i,4]< 1.1*area:
            selected_label=i

    img_selected=np.zeros(img.shape, dtype='uint8')
    img_selected[img_labels==selected_label]=255
    
    return(img_selected)

def aprox_circ(img_hole):
    '''
    This function return the circle festure of a contour approximating it.

    Parameters
    ----------
    img_hole : Image of the label that will be approximated

    Returns
    -------
    radio_hole: radio of the approximated circle
    
    hole_center: hole of the approximated circle
    
    external_hole_pts: contour of the label not yet approximated

    '''
    
    # Gettingexternal contour of hole image
    cnt_hole=cv2.findContours(img_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    external_hole_pts=cnt_hole[0]
    
    # Fitting a circunference in contour
    inliers= ransac.ransac_circunf(external_hole_pts,100) #finding best inliers randomly
    circABC=lms.circunf(inliers);   #least minimum square on the inliners
    A,B,C = circABC

    radio_hole=int(np.sqrt(A**2+B**2-4*C)/2)
    hole_center=(int(-B/2),int(-A/2))
    
    return(radio_hole,hole_center,external_hole_pts)

def keyway_fn(img_hole,hole_center,radio_hole):
    '''
    This function draw a black circle into the hole_image,
    check for kayway label and find it

    Parameters
    ----------
    img_hole: image that will be cutted
    
    hole_center: center of the cutting circunference
        
    radio_hole: radio of the cutting circunference

    Returns
    -------
    keyway_img: label of the keyway
    
    keyway_centroid: cetroid of keyway label based on area calculus

    '''
    y_hole=hole_center[0]
    x_hole=hole_center[1]
    
    # Cutting the hole image
    cv2.circle(img_hole, (x_hole,y_hole),radio_hole+2,(0,0,0),-1)
   
    # Checking for labels and find the keyway label    
    num_labels, img_labels=cv2.connectedComponents(img_hole)
    list_img=[]

    for num_label in range(1, num_labels):  # whithot label 0 (=background)
        img_bin_object = np.zeros_like(img_labels, np.uint8)
        img_bin_object[img_labels == num_label] = 1 # Making the number of every label = 1
        list_img.append(img_bin_object) # storing the labels

    keyway_img=list_img[0]
    keyway_y, keyway_x = bf.centroid(keyway_img) # using white pixels (unit mass) to do calculus
    keyway_centroid=(keyway_y, keyway_x)
    
    return(keyway_img,keyway_centroid)

def width_fn(keyway_img):
    '''
    This function retun the keyway width using min rectangular area that fits
    
    Parameters
    ----------
    keyway_img : label of the keyway

    Returns
    -------
    real_width : maximum dimension of min rectangular area
        
    external_key_pts: contour of the keyway

    '''
    
    # Getting external contour of keyway image
    cnt_key=cv2.findContours(keyway_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    external_key_pts=cnt_key[0]
    
    # Fitting a rectangle into the keyway
    (_,(width,height),_)=cv2.minAreaRect(external_key_pts)
    real_width=max(width,height)

    return real_width,external_key_pts

def distance_fn(hole_center,external_key_pts):
    '''
    This function find the rect that intersect the extreme of keyway and evaluate the distance

    Parameters
    ---------- 
    
    external_key_pts : contour of the keyway
    
    hole_center : center of the internal hole

    Returns
    -------
    distance : distance between point and line

    '''
    
    # Fitting a line in contour
    inliers= ransac.ransac_recta(external_key_pts,100) 
    rectABC=lms.recta(inliers);   
    a,b,c = rectABC
    
    y_hole=hole_center[0]
    x_hole=hole_center[1]
    
    # Distance between line and point
    distance = (abs(a*x_hole + b*y_hole + c)) / (np.sqrt( a*a + b*b))
    
    return distance

def rotation_fn(hole_center,keyway_centroid):
    '''
    This function evaluate the rotation with rispect vertical axes using artan function

    Parameters
    ----------
    hole_center : centre of hole
   
    keyway_centroid : centre of keyway

    Returns
    -------
    rotation : rotation

    '''
    keyway_y=keyway_centroid[0]
    keyway_x=keyway_centroid[1]
    y_hole=hole_center[0]
    x_hole=hole_center[1]
    
    # Rotation using arctangent function
    rotation=(np.arctan2(keyway_y-y_hole,keyway_x-x_hole))*180/3.14+90
    
    # Adjusting the rotation with respect new coordinate system
    if keyway_y >= y_hole and keyway_x <= x_hole:
        rotation=rotation-360
    else:
        rotation=rotation
        
    return rotation

def teeth_fn(img_gear):
    '''
    This function evaluate the avarage external circunference and through it cut the gear 

    Parameters
    ----------
    img_gear : label of the gear

    Returns
    -------
    radio_gear: radio of avarage circle
    
    num_teeht: number of labelled teeth

    '''
    
    # Getting external contour from gear image
    cnt_gear = cv2.findContours(img_gear, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    list_contours=sorted(cnt_gear, key=np.sum,reverse=False) #to sort the list of contours in ascending order based on pixels sum
    external_gear_pts=list_contours[-1] #taking the biggest white object contour

    # Fitting a circunference in contour
    circABC=lms.circunf(external_gear_pts);
    A,B,C = circABC
    
    radio_gear=int(np.sqrt(A**2+B**2-4*C)/2)
    x_hole=int(-A/2)
    y_hole=int(-B/2)
    
    # Cutting the gear with a black circle
    cv2.circle(img_gear, (x_hole,y_hole),radio_gear,(0,0,0),-1)       
      
    # Finding the number of remaining lables
    num_labels, img_label = cv2.connectedComponents(img_gear)
    num_teeth=num_labels-1
    
    return(radio_gear,num_teeth)

def print_results(img, num_teeth, radio_hole, radio_gear , angle, width, distance,hole_center,keyway_centroid):
    '''
    This function print in up left corner: number of teeths, angle ad radius.
    This function print in down left corner: width and distance.

    '''
    row,colum=img.shape
    img_color_output = cv2.merge([img,img,img])
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(img_color_output, (hole_center[1],hole_center[0]), (keyway_centroid[1],keyway_centroid[0]), (255,0,0),2)
    cv2.line(img_color_output, (hole_center[1],hole_center[0]), (hole_center[1],0),(255,0,0), 2)
    cv2.circle(img_color_output,(hole_center[1],hole_center[0]), radio_hole,(255,0,0),2)
    cv2.circle(img_color_output,(hole_center[1],hole_center[0]), radio_gear,(255,0,0),2)
    cv2.putText(img_color_output,'Number of teeth: '+str(num_teeth),(10,30),font,1,(255,0,0),2)
    cv2.putText(img_color_output,'Angle: '+str(round(angle,1)),(10,60),font,1,(255,0,0),2)
    cv2.putText(img_color_output,'Radius: '+str(radio_hole),(10,90),font,1,(255,0,0),2)
    cv2.putText(img_color_output,'Width: '+str(round(width)),(10,row-60),font,1,(255,0,0),2)
    cv2.putText(img_color_output,'Distance: '+str(round(distance)),(10,row-30),font,1,(255,0,0),2)
    return img_color_output

# =============================================================================
# INPUT SELECTION
# =============================================================================

# img_title = 'samples/engranaje1.png' 
img_title = 'samples/engranaje2.png' #angle 
# img_title = 'samples/engranaje3.png' 
# img_title = 'samples/engranaje4.png' 
# img_title = 'samples/engranaje5.png'
# img_title = 'samples/engranaje6.jpg'
# img_title = 'samples/engranaje7.png' #more objects

# =============================================================================
# SCRIPT
# =============================================================================

img = cv2.imread(img_title, 0)

AREA_HOLE= 17500
AREA_GEAR= 86000

# Finding the white gear with BINARY_INV because, in origin, it is black
img_bin_inv=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
img_gear=area_selection(img_bin_inv, AREA_GEAR)

# Finding the white hole on the previous gear image INVERTING it
img_bin=cv2.bitwise_not(img_gear)
img_hole=area_selection(img_bin, AREA_HOLE)

# Getting general info white hole
radio_hole,hole_center,external_hole_pts=aprox_circ(img_hole)

# Finding the white keyway from hole image
keyway_img,keyway_centroid=keyway_fn(img_hole,hole_center,radio_hole)

# Getting information from keyway image
width,external_key_pts=width_fn(keyway_img)
distance=distance_fn(hole_center,external_key_pts)
rotation=rotation_fn(hole_center,keyway_centroid)

# Finding number of teeths from gear image
radio_gear,num_teeth=teeth_fn(img_gear)

# Getting results
img_color_output=print_results(img, num_teeth, radio_hole, radio_gear , rotation, width, distance,hole_center,keyway_centroid)

# Printing results
plt.imshow(img_color_output)
plt.title('Results')
plt.axis('off')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))