# SWCPAN
SWCPAN: A HYBRID NETWORK OF SWIN TRANSFORMER AND CNN FOR PAN-SHARPENING

## Introduction
SWCPAN comprises a shallow feature extraction module, a deep feature extraction module and an image reconstruction module. Specifically, the shallow feature extraction module is based on convolutional layers, whereas the deep feature extraction module is composed of several Swin Transformer blocks (STB) that exploit the local attention and shifted windowing scheme of Swin Transformer to improve the quality of the generated images. Experimental results on GaoFen-2 and WorldView-3 datasets demonstrate that our SWCPAN outperforms most state-of-the-art methods in several metrics.
![overview](/Figs/overview.jpg)

<img src="/Figs/STB.jpg" width="49%"> <img src="/Figs/IRM.jpg" width="49%">

## Qualitative results

### GaoFen-2

### WorldView-3

