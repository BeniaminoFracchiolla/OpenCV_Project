# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:13:20 2023

@author: benny
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def get_random_points( puntos,N):
    '''Elige N puntos aleatoriamente del conjunto (xPixelsCont,yPixelsCont)
    Devuelve dos vectores: xSample con las coord x y ySample con las coord y'''
    samples=[]

    numPixelsCont=len(puntos)
    #N numero de puntos aleatorios
    for k in range(N):
        indexRand=random.randrange(0,numPixelsCont)
        samples.append( puntos[indexRand] )
    return np.array(samples)

def print_results(img, num_teeth, radius, angle):
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Number of teeth:'+str(num_teeth),(10,30),1,(255,0,0),2,font)
    cv2.putText(img,'Radius:'+str(radius),(10,40),1,(255,0,0),2,font)
    cv2.putText(img,'Angle:'+str(num_teeth),(10,50),1,(255,0,0),2,font)
    return 

def draw_circunf_ABC(img, circABC,color, grosor=1):
    if len(img.shape)==2:#si img es de niveles de gris
        img_color=cv2.merge([img,img,img])
    else:
        img_color=img.copy()
    [A,B,C]=circABC
    radio= int (np.sqrt(A**2+B**2-4*C)/2)
    centro= (int(-A/2), int(-B/2));
    cv2.circle(img_color, centro, radio, color, grosor)
    plt.imshow(img_color)
    plt.axis('off')
    plt.title("circunf")
    plt.show()
    return img_color

def lms_circunf(puntos):
    '''minimos cuadrados circunferencia calculando la pseudoinversa
    Devuelve los parametros A,B,C de la circunferencia x2+y2+Ax+By+C=0
    que ajusta los puntos'''

    n=len(puntos)
    A=np.ones( (n,3), np.float32)
    
    for i in range(n):
        A[i][0]= puntos[i][0][0]
        A[i][1]= puntos[i][0][1]

    b=np.ones( (n,1), np.float32)
    for i in range(n):
        b[i][0] = - puntos[i][0][0]**2 - puntos[i][0][1]**2 


    # The pseudo-inverse of a matrix A, denoted A^+, is defined as: 
    # the matrix that ‘solves’ [the least-squares problem] Ax = b,
    # i.e., if \bar{x} is said solution, then A^+ is that matrix such that x = A^+ b.        
    
    pseudo_inv_A= np.linalg.pinv(A)

    [A,B,C]=pseudo_inv_A@b;

    return [A,B,C]

def inliers_circunf( ptos_cnt,A,B,C,tolerancia):
    ''' Devuelve inliers para circunferencia x2+y2+ Ax+By+C=0'''
    radio= np.sqrt(A**2+B**2-4*C)/2
    centro= ((-A/2), int(-B/2));
    #franja de consenso (R1,R2)
    R1=radio-tolerancia;
    R2=radio+tolerancia;
    inliers=[]
    num_ptos=len(ptos_cnt)
    num_inliers=0;
    for k in range(num_ptos):
        x=ptos_cnt[k][0][0]
        y=ptos_cnt[k][0][1]        
        dist_centro = np.sqrt((x-centro[0])**2 + (y-centro[1])**2);

        if  R1 < dist_centro < R2:  #Si el pixel esta en la corona 
            num_inliers = num_inliers + 1;
            inliers.append( ptos_cnt[k] )
    
    return np.array(inliers)

def ransac_circunf(puntos,num_max_iter):
    '''puntos(x,y) contiene las x (columnas)y las y (filas) de los pix de cont
    num_max_iter numero máximo de iteraciones
    mejor_inliers son los puntos del mejor modelo encontrado
    inliers hay que pasarlos LMS para afinar circunferencia
    devuelve mejor_inliers
    '''

    tolerancia=1; # anchura de la corona circular donde estan los inliers
    num_inliers=0;
    mejor_num_inliers=0; #el numero de inliers más alto q se ha encontrado

    for i in range(num_max_iter):
        samples =get_random_points(puntos, 3)#cogemos 3 puntos

        #circunferencia que pasa por estos tres puntos
        [A, B, C] = lms_circunf(samples)
       
        inliers = inliers_circunf( puntos, A,B,C, tolerancia)

        num_inliers=len(inliers);
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers=num_inliers
            mejor_inliers=inliers

    return mejor_inliers
