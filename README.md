# ERP2CPP
Python implementation for converting 360° ERP format image/video to CPP format.

The code was adapted from the C++ version of CPP-PSNR at https://github.com/I2-Multimedia-Lab/360-video-experimental-platform/blob/master/cpp-psnr/cpppsnr_metric.cpp

I replace the for-loop with vectorize operation to speed up convertion, but the readability of the code may become awful.

## Requirements
The only dependency of this code is Numpy.

## Usage:
The input should be ERP image of (H,W,C) or video of (T,H,W,C) in Numpy array.
```
from erp_cpp import ERP2CPP
import skimage.io
img=skimage.io.imread('erp.png')
erp2cpp=ERP2CPP('lanczos') # or ERP2CPP('nearest')
cpp=erp2cpp(img)
skimage.io.imsave('cpp.png', cpp)
```
```
from erp_cpp import ERP2CPP
import skvideo.io
video=skvideo.io.vread('erp.mp4')
erp2cpp=ERP2CPP('lanczos')
cpp=erp2cpp(video)
skvideo.io.vwrite('cpp.mp4', cpp)
```
