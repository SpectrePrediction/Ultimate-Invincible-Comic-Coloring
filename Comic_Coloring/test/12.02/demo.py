import models as md
import numpy as np
import tensorflow as tf
import cv2
import os
import losses as loss

# input_img = list()
# img = cv2.imread('test1.jpg', 0)
# input_img.append(img)
# input_img.append(img)
# input_img.append(img)
# input_img.append(img)
# input_img = np.array(input_img)
# input_img = input_img.reshape((1, img.shape[0], img.shape[1], 4))
# # input_img = input_img.astype('float32') / 255.0
# print(input_img.shape)


input_shape = (16, 512, 512, 1)
x = tf.random.normal(input_shape)
y = tf.random.normal(input_shape)
print(loss.blur_dist_loss(x, y).shape)

# input_shape = (32, 128, 128, 32)
# local = tf.random.normal(input_shape)
# newimg = input_img[0]
# cv2.imshow("t", newimg)
# cv2.waitKey(0)

# input_x = np.zeros((32, 256, 256, 1))
# print(input_x.shape)
# test = md.DownCNNBlock(512, block_num=4, mid_channel=32)
test = md.ChangedUNetG()# DownCNNBlock(512)
# test = md.Conv2DSubPixel(2)
# test = md.Up(32)
# test = md.GroupConv2D(64, 3, groups=32)

pred = test(x)

print(pred.shape)

# print(test.summary())