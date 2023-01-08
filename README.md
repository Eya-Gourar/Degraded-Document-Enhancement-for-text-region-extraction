# Degraded Document Enhancement for text region extraction

This repo contains different approaches to binarize degraded documents using both simple thresholding techniques, such as : Otsu, Integral Images & Contrast Enhancement methods, and deep Learning threshold-based methods as in the use of Transformers in Binarization context and GANs. Then test OCR modules *pytesseract* , *keras_ocr* & *easyocr* on both original and binarized samples to see what output comes out.

Folder Structure 
================

> an overview of the most important parts of this repository.
### Tree Directory layout
    .
    ├── Binarization                # folder containing different threshold-based binarization methods
    │   ├── TransformerEnhancer     # Transformers method
    │   ├── DEGAN                   # GANs method
    │   ├── IntegralImages.py       # integral images method .py
    │   ├── Otsu.py                 # Otsu method .py
    │   └── ContrastEnhancement.py   # Contrast enhancement based method .py
    ├── OCRs.ipynb                  # test OCR techniques Keras-OCR VS EasyOCR VS PYTESSERACT 
    ├── demo                        # >> where you can test our code :D
    │   ├── degraded                # DIBCO images samples
    │   ├── Output                  # Different output of binarization methods
    │   └── demo.ipynb              # the demo .. duh
    ├── requirements.txt
    └── README.md

Introduction 
================

As the optical character recognition (OCR) techniques have become widely available, a crucial first step for OCR remains document image binarization. Image binarization sets the gray values of pixels to 0 or 255, creating a black and white image. Although document image binarization has been studied for many years, extracting clear characters from degraded document images is still a challenging problem.

Threshold-based methods have been researched widely because of their briefness, efficiency, and easy comprehension.

## 1. Otsu :

Otsu’s method : [A Threshold Selection Method from Gray-Level Histograms](https://cw.fel.cvut.cz/b201/_media/courses/a6m33bio/otsu.pdf) is a variance-based technique to find the threshold value where the weighted variance between the foreground and background pixels is the least.

## 2. Integral Images :

A modified version of Sauvola Method with the use Integral Images. We took inspiration from the paper : [Efficient Implementation of Local Adaptive Thresholding Techniques Using Integral Images](https://dll.seecs.nust.edu.pk/wp-content/uploads/2020/06/Efficient-implementation-of-local-adaptive-thresholding-techniques-using-integral-images.pdf) where it's shown how Sauvola method can be made almost as much efficient as Otsu Binarization, independent of the window size and without having any impact on the method's quality.

## 3. Contrast Enhancement :

The grayscale enhancement method can effectively be used to widen the contrast between pixels so that the foreground can be separated from the background of an image with non-uniform illumination, bleed-through, and variable background. Reference [Binarization of degraded document images based on contrast enhancement](https://rdcu.be/c2ibU)

## 4. Enhancer Transformer :

Pytorch implementation of the paper [DocEnTr: An End-to-End Document Image Enhancement Transformer](https://arxiv.org/abs/2201.10252). A scalable auto-encoder that uses vision transformers in its encoder and decoder parts. The degraded image is first divided into patches before entering to the encoder part. During encoding, the patches are mapped to a latent representation of tokens, where each token is associated with a degraded patch.
Then, the tokens are passed to the decoder that outputs the enhanced version of patches.

## 5. DE-GAN :

This is an implementation for the paper [DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement](https://ieeexplore.ieee.org/document/9187695)<br>
DE-GAN is a conditional generative adversarial network designed to enhance the document quality before the recognition process. It could be used for document cleaning, binarization, deblurring and watermark removal. The weights are available to test the enhancement. 

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow">Binarization methods</th>
    <th class="tg-c3ow">PSNR (dB)</th>
    <th class="tg-c3ow">F-measure (%)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="2">0</td>
    <td class="tg-c3ow" rowspan="2"><br>Otsu</td>

  </tr>
  <tr>
    <td class="tg-c3ow">15.17</td>
    <td class="tg-c3ow">78.48</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">1</td>
    <td class="tg-c3ow" rowspan="2"><br>Integral Images</td>

  </tr>
  <tr>
    <td class="tg-c3ow">15.16</td>
    <td class="tg-c3ow">80.66</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">2</td>
    <td class="tg-c3ow" rowspan="2"><br>Contrast Enhancement</td>

  </tr>
  <tr>
    <td class="tg-c3ow">15.86</td>
    <td class="tg-c3ow">82.75</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">3</td>
    <td class="tg-c3ow" rowspan="2"><br>Enhancer Transformer</td>
  </tr>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
    <tr>
    <td class="tg-c3ow" rowspan="2">3</td>
    <td class="tg-c3ow" rowspan="2"><br>GAN Enhancer</td>
  </tr>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
</tbody>
</table>

About the Code :
================


### Data :

We have gathered some sample degraded images of the DIBCO 2009 and H-DIBCO datasets and organized them in one folder  **demo/degraded/**. 
We performed the proposed different techniques and saved outputs in their respective folders **demo/Output/**.

### Demo :
In this demo, we show how we can use our proposed techniques to binarize degraded images of your choice, this is detailed in the file named demo.ipynb for simplicity we make it a jupyter notebook where you can chose your desired binarization technique and visualize your results.

### Requirements
- install requirements.txt
- for the transformer model, download the best pre-trained model that has the highest PSNR from this [link](https://drive.google.com/file/d/1FKXAS8BetcB2pCwkOTNHIX4Rj5-tq-Ep/view) and store it into **weights** directory after creating it under **Binarization/TransformerEnhancer/** .
- then, download the trained weights to directly use the model for the GAN document enhancement, it is important to save these weights in the subfolder named **weights**, in the **Binarization/DEGAN** folder. Here is the [link](https://drive.google.com/file/d/1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-/view?usp=sharing) to download the weights.

# Conclusion
There should be no bugs in this code, but if there is, we are sorry for that :') !!

