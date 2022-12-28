#libraries
import cv2
import os

def otsu(img) :

    #reading the image
    filename = "demo/degraded/demo.jpg"
    cv2.imwrite(filename, img)
    a = cv2.imread(filename)

    #making 3 grayscale images 
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    d = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    #otsu thresholding
    ret2,th= cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th