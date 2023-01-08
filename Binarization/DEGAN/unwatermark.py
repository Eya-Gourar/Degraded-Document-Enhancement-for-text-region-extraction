
#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
from PIL import Image
import random
from utils import *
from models.models import *
import cv2


def unwatermark(img) :
    input_size = (256,256,1)
    import os
    cwd = os.getcwd()
    #files = os.listdir(cwd)

    model_path = cwd + "/Binarization/DEGAN/weights/watermark_rem_weights.h5"
    generator.load_weights(model_path)


    deg_image_path = cwd + "/demo/degraded/demo.jpg"
    cv2.imwrite(deg_image_path, img)
    deg_image = Image.open(deg_image_path)# /255.0
    deg_image = deg_image.convert('L')
    deg_image.save(cwd+ '/demo/decurr_image.png')

    test_image = plt.imread(cwd + '/demo/decurr_image.png')


    h =  ((test_image.shape [0] // 256) +1)*256 
    w =  ((test_image.shape [1] // 256 ) +1)*256

    test_padding=np.zeros((h,w))+1
    test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

    test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
    predicted_list=[]
    for l in range(test_image_p.shape[0]):
        predicted_list.append(generator.predict(test_image_p[l].reshape(1,256,256,1)))

    predicted_image = np.array(predicted_list)#.reshape()
    predicted_image=merge_image2(predicted_image,h,w)

    predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
    predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])


    save_path = cwd+ '/demo/Output/ganOutput.png'
    plt.imsave(save_path, predicted_image,cmap='gray')

    result = plt.imread(save_path)

    return result


