#libraries
import numpy as np
from numpy import linalg
import cv2
from math import sqrt
from math import atan
from math import pow

def II(img) :

    #reading the image

    filename = "demo/degraded/demo.jpg"
    cv2.imwrite(filename, img)
    a = cv2.imread(filename)

    #making 3 grayscale images 
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    d = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    #k-value range [0.1-0.5] can be changed depending on image
    k=0.2

    #r-value can be changed depending on image
    r=128

    #neighbouring window size can be changed depending on image
    window=35

    h,w=b.shape

    th=0
    value=[]
    threshold=[]
    m=0
    var=0
    std=0
    b,c=cv2.integral2(b)

    h,w=c.shape

    #Sauvola Method using integrl images
    for i in range(1, w , 1):
        for j in range(1, h , 1):
            if(j>(h-int(window/2)) and i>(w-int(window/2))):
                m = (b[j, i] + b[j - int(window/2), i - int(window/2)] - b[
                    j , i - int(window/2)] - b[j - int(window/2), i]) / (window * window)
                s = (c[j, i] + c[j - int(window/2), i - int(window/2)] - c[
                    j, i - int(window/2)] - c[j - int(window/2), i]) / (window * window)
            elif (i > (w - int(window/2)) and j < (h - int(window/2))):
                m = (b[j + int(window/2), i] + b[j - int(window/2), i - int(window/2)] - b[
                    j + int(window/2), i - int(window/2)] - b[j - int(window/2), i]) / (window * window)
                s = (c[j + int(window/2), i] + c[j - int(window/2), i - int(window/2)] - c[
                    j + int(window/2), i - int(window/2)] - c[j - int(window/2), i]) / (window * window)
            elif(j>(h-int(window/2)) and i<(w-int(window/2))):
                m = ( b[j,i+int(window/2)]+b[j-int(window/2),i-int(window/2)]-b[j,i-int(window/2)]-b[j-int(window/2),i+int(window/2)])/(window*window)
                s = (  c[j,i+int(window/2)]+c[j-int(window/2),i-int(window/2)]-c[j,i-int(window/2)]-c[j-int(window/2),i+int(window/2)])/(window*window)
            elif(j<(h-int(window/2) ) and i<(w-int(window/2))):
                m=(b[j+int(window/2),i+int(window/2)]+b[j-int(window/2),i-int(window/2)]-b[j+int(window/2),i-int(window/2)]-b[j-int(window/2),i+int(window/2)])/(window*window)
                s=(c[j+int(window/2),i+int(window/2)]+c[j-int(window/2),i-int(window/2)]-c[j+int(window/2),i-int(window/2)]-c[j-int(window/2),i+int(window/2)])/(window*window)
            var = ((s)- (pow((m), 2)))/(window*window)
            std = sqrt(abs(var))
            T = m * (1 + (k * ((std / r) - 1)))
            threshold.append(T)


    h1,w1=d.shape

    #setting pixel values based on computed threshold
    for i in range(0, w1, 1):
         for j in range(0, h1, 1):

            if (d[j, i] <= threshold[th]):
                 d[j, i] = 0
            else:
                 d[j, i] = 255
            th=th+1

    for i in range(0, int(window/2), 1):
        for j in range(0, h1, 1):
            d[j,i]=255
    for i in range(0, w1, 1):
        for j in range(0, int(window/2), 1):
            d[j,i]=255

    width = a.shape[1]
    height = a.shape[0]
    dim = (width, height)
    # resize image
    resized = cv2.resize(d, dim)
    return resized