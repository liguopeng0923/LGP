import cv2
import numpy as np
from mmcv.image import imread,imwrite
img = imread("/remote-home/liguopeng/object_detection/remote_sensing/cvpr2023/oriented_rcnn/DAG/images/P0003__1024__0___0.png")
blur =  cv2.GaussianBlur(img,(3,3),0)
imwrite(blur,"test.png")