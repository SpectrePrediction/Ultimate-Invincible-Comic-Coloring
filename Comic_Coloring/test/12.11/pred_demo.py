import tensorflow as tf
import losses as loss
import datautil as util
import models as md
import numpy as np
import cv2


gray = cv2.imread("gray5.jpg")
color = cv2.imread("color5.jpg")
img_size = 224

gray = util.img_padding()(gray)
gray = util.img_resize((img_size, img_size))(gray)

color = util.img_padding()(color)
color = util.img_resize((img_size, img_size))(color)
# color = util.img_BGR2YUV()(color)

gray = gray.reshape((1, gray.shape[0], gray.shape[1], gray.shape[2]))
color = color.reshape((1, color.shape[0], color.shape[1], color.shape[2]))

gray = gray.astype('float32') / 255.0
color = color.astype('float32') / 255.0

model = md.ChangedUNetG(out_channel=2)


model.build((1, img_size, img_size, 3))
model.load_weights("my_model_test509.h5", by_name=True)
pred = model(gray)
pred = pred.numpy()

y = gray[:, :, :, 0]
img = (cv2.merge((y[0], pred[0])) * 255).astype("uint8")
img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
# colornew = (color * 255).astype("uint8")[0]
# colornew = cv2.cvtColor(colornew, cv2.COLOR_YUV2BGR)
colorold = (color * 255).astype("uint8")[0]
# colorold = cv2.cvtColor(colorold, cv2.COLOR_YUV2BGR)

grayold = (gray * 255).astype("uint8")[0]

img = np.hstack([img, grayold, colorold])

cv2.imwrite(f"model_test_5_509.jpg", img)
