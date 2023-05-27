# SWCPAN
SWCPAN: A HYBRID NETWORK OF SWIN TRANSFORMER AND CNN FOR PAN-SHARPENING

## Introduction
SWCPAN comprises a shallow feature extraction module, a deep feature extraction module and an image reconstruction module. Specifically, the shallow feature extraction module is based on convolutional layers, whereas the deep feature extraction module is composed of several Swin Transformer blocks (STB) that exploit the local attention and shifted windowing scheme of Swin Transformer to improve the quality of the generated images. Experimental results on GaoFen-2 and WorldView-3 datasets demonstrate that our SWCPAN outperforms most state-of-the-art methods in several metrics.
![overview](/Figs/overview.jpg)

<img src="/Figs/STB.jpg" width="49%"> <img src="/Figs/IRM.jpg" width="49%">

## Qualitative results

### GaoFen-2
<img src="/Qualitative/gf2/1bicubic.png" width="13%"> <img src="/Qualitative/gf2/1PNN.png" width="13%"> <img src="/Qualitative/gf2/1PANNET.png" width="13%"> <img src="/Qualitative/gf2/1MSDCNN.png" width="13%"> <img src="/Qualitative/gf2/1PANFORMER.png" width="13%"> <img src="/Qualitative/gf2/1SWINPAN.png" width="13%"> <img src="/Qualitative/gf2/1groudtruth.png" width="13%">

<img src="/Qualitative/gf2/2bicubic.png" width="13%"> <img src="/Qualitative/gf2/2PNN.png" width="13%"> <img src="/Qualitative/gf2/2PANNET.png" width="13%"> <img src="/Qualitative/gf2/2MSDCNN.png" width="13%"> <img src="/Qualitative/gf2/2PANFORMER.png" width="13%"> <img src="/Qualitative/gf2/2SWINPAN.png" width="13%"> <img src="/Qualitative/gf2/2groudtruth.png" width="13%">

<img src="/Qualitative/gf2/3bicubic.png" width="13%"> <img src="/Qualitative/gf2/3PNN.png" width="13%"> <img src="/Qualitative/gf2/3PANNET.png" width="13%"> <img src="/Qualitative/gf2/3MSDCNN.png" width="13%"> <img src="/Qualitative/gf2/3PANFORMER.png" width="13%"> <img src="/Qualitative/gf2/3SWINPAN.png" width="13%"> <img src="/Qualitative/gf2/3groudtruth.png" width="13%">

### WorldView-3

<img src="/Qualitative/wv3/1bicubic.png" width="13%"> <img src="/Qualitative/wv3/1PNN.png" width="13%"> <img src="/Qualitative/wv3/1PANNET.png" width="13%"> <img src="/Qualitative/wv3/1MSDCNN.png" width="13%"> <img src="/Qualitative/wv3/1PANFORMER.png" width="13%"> <img src="/Qualitative/wv3/1SWINPAN.png" width="13%"> <img src="/Qualitative/wv3/1groudtruth.png" width="13%">

<img src="/Qualitative/wv3/2bicubic.png" width="13%"> <img src="/Qualitative/wv3/2PNN.png" width="13%"> <img src="/Qualitative/wv3/2PANNET.png" width="13%"> <img src="/Qualitative/wv3/2MSDCNN.png" width="13%"> <img src="/Qualitative/wv3/2PANFORMER.png" width="13%"> <img src="/Qualitative/wv3/2SWINPAN.png" width="13%"> <img src="/Qualitative/wv3/2groudtruth.png" width="13%">

<img src="/Qualitative/wv3/3bicubic.png" width="13%"> <img src="/Qualitative/wv3/3PNN.png" width="13%"> <img src="/Qualitative/wv3/3PANNET.png" width="13%"> <img src="/Qualitative/wv3/3MSDCNN.png" width="13%"> <img src="/Qualitative/wv3/3PANFORMER.png" width="13%"> <img src="/Qualitative/wv3/3SWINPAN.png" width="13%"> <img src="/Qualitative/wv3/3groudtruth.png" width="13%">
