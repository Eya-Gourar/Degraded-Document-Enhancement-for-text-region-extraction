import torch
from vit_pytorch import ViT
from models.binae import BINMODEL
import numpy as np
from einops import rearrange
import cv2
import matplotlib.pyplot as plt

THRESHOLD = 0.5 ## binarization threshold after the model output

SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
patch_size = 8 ## choose your desired patch size [8 or 16], depending on the model you want to use

image_size =  (SPLITSIZE,SPLITSIZE)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model settings
ENCODERLAYERS = 6
ENCODERHEADS = 8
ENCODERDIM = 768

v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = ENCODERDIM,
    depth = ENCODERLAYERS,
    heads = ENCODERHEADS,
    mlp_dim = 2048
)
model = BINMODEL(
    encoder = v,
    masking_ratio = 0.5,   # __ doesnt matter for binarization
    decoder_dim = ENCODERDIM,      
    decoder_depth = ENCODERLAYERS,
    decoder_heads = ENCODERHEADS       
)

model = model.to(device)

model_path = "weights/best-model_8_2018base_256_8(2).pt"
model.load_state_dict(torch.load(model_path))

def split(im,h,w):
    patches=[]
    nsize1=SPLITSIZE
    nsize2=SPLITSIZE
    for ii in range(0,h,nsize1): #2048
        for iii in range(0,w,nsize2): #1536
            patches.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
    return patches 

def merge_image(splitted_images, h,w):
    image=np.zeros(((h,w,3)))
    nsize1=SPLITSIZE
    nsize2=SPLITSIZE
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
            ind += 1
    return image  


def Enhancer(img):

    filename = "demo/degraded/demo.jpg"
    cv2.imwrite(filename, img)
    deg_image = cv2.imread(filename)

    ## Split the image intop patches, an image is padded first to make it dividable by the split size
    h =  ((deg_image.shape[0] // 256) +1)*256 
    w =  ((deg_image.shape[1] // 256 ) +1)*256
    deg_image_padded=np.ones((h,w,3))
    deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:]= deg_image
    patches = split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])

    ## preprocess the patches (images)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    out_patches=[]
    for p in patches:
        out_patch = np.zeros([3, *p.shape[:-1]])
        for i in range(3):
            out_patch[i] = (p[:,:,i] - mean[i]) / std[i]
        out_patches.append(out_patch)


    result = []
    for p in out_patches:
        p = np.array(p, dtype='float32')
        train_in = torch.from_numpy(p)



        with torch.no_grad():
            train_in = train_in.view(1,3,SPLITSIZE,SPLITSIZE).to(device)
            _ = torch.rand((train_in.shape)).to(device)


            loss,_, pred_pixel_values = model(train_in,_)

            rec_patches = pred_pixel_values

            rec_image = torch.squeeze(rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size))

            impred = rec_image.cpu().numpy()
            impred = np.transpose(impred, (1, 2, 0))

            for ch in range(3):
                impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

            impred[np.where(impred>1)] = 1
            impred[np.where(impred<0)] = 0
        result.append(impred)

    clean_image = merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
    clean_image = clean_image[:deg_image.shape[0], :deg_image.shape[1],:]
    clean_image = (clean_image>THRESHOLD)*255
