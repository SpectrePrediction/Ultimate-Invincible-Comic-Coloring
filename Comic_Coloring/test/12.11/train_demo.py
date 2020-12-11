import tensorflow as tf
import losses as loss
import datautil as util
import models as md
import numpy as np
import cv2


class BlurLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        
        return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))) /\
           tf.abs(-tf.reduce_mean(tf.keras.losses.cosine_similarity(y_pred, y_true, 1)) + 0.001)

# def my_blur_loss():
#     def blur_dist_loss(y_true, y_pred):
#         return loss.blur_dist_loss(tf.constant(y_true), y_pred)

#     return blur_dist_loss

# loss_obj = tf.keras.losses.CosineSimilarity(axis=1)
loss_obj = BlurLoss()

def blur_loss(model, x, y):
    pred = model(x)

    return loss_obj(y_true=y, y_pred=pred)
    # return loss.cosine_similarity(y, pred)


def main():
    temp_epoch = -1
    dr = util.DataReader(r"train.txt", [
            util.img_padding(),
            util.img_resize((224, 224)),
            util.img_BGR2YUV()
        ],
     read_data_cache=0,
     batch_size=64,
     is_completion=True
     )

    model = md.ChangedUNetG(out_channel=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


    @tf.function    
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss_value = loss_obj(labels, predictions)# tf.reduce_mean(tf.keras.losses.MSLE(labels, predictions))
            # print(loss_value)



        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss_value, predictions

    
    

    for epoch, color, gray in dr:
        gray = gray.astype('float32') / 255.0
        color = color.astype('float32') / 255.0
        y = color[:, :, :, 0]
        u = color[:, :, :, 1]
        v = color[:, :, :, 2]
  
        u = u.reshape((1, *u.shape))
        v = v.reshape((1, *v.shape))

        loss_img = cv2.merge((u, v))
        loss_img = loss_img[0]
        
        # cv2.imshow("t", color[:, :, :, 0][0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # color_YUV = tf.image.rgb_to_yuv(color)
        # cv2.imshow("tst", gray[0])
        # cv2.imshow("clor", color[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        loss_value, pred = train_step(gray, loss_img)
        step = optimizer.iterations.numpy()

        print(f"Step: {step}, Initial Loss: {loss_value.numpy()}, epoch{epoch}")

        if step % 100 == 0:
            pred = model(gray)
            pred = pred.numpy()

            y = gray[:, :, :, 0]
            img = (cv2.merge((y[0], pred[0])) * 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            # colornew = (color * 255).astype("uint8")[0]
            # colornew = cv2.cvtColor(colornew, cv2.COLOR_YUV2BGR)
            colorold = (color * 255).astype("uint8")[0]
            colorold = cv2.cvtColor(colorold, cv2.COLOR_YUV2BGR)

            grayold = (gray * 255).astype("uint8")[0]

            img = np.hstack([img, grayold, colorold])

            cv2.imwrite(f"./image/{step}.jpg", img)

            # cv2.imshow("color", color[0])
            # cv2.imshow("colornew", colornew)
            # cv2.imshow("pred", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        if temp_epoch != epoch:
            temp_epoch = epoch
            model.save_weights('my_model_test.h5', save_format='h5')


if __name__ == '__main__':
    main()
