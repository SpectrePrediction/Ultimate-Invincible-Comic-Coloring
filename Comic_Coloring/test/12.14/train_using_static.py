import tensorflow as tf
import losses as loss
import datautil as util
import models as md
import numpy as np
import cv2
import conf as cf
from sklearn.utils import shuffle


def main():
    temp_epoch = -1
    dr = util.DataReader(cf.image_txt_path, [
            util.img_padding(),
            util.img_resize((cf.img_size, cf.img_size)),
            util.img_BGR2YUV()
        ],
        read_data_cache=cf.read_data_cache,
        batch_size=cf.batch_size,
        is_completion=cf.reader_is_completion,
        is_shuffle=cf.reader_is_shuffle,
        is_show_progress=cf.reader_is_show_progress
    )

    model = md.ChangedUNetG(**cf.model_args)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=cf.learning_rate)
    loss_obj = loss.CosineSimilarityEuclideanLoss()

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss_value = loss_obj(labels, predictions)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss_value, predictions

    loss_list = util.CacheList(cf.reduce_lr_static_args['patience'])
    for epoch, color, gray in dr:
        gray = gray.astype('float32') / 255.0
        color = color.astype('float32') / 255.0
        color, gray = shuffle(color, gray)

        u = color[:, :, :, 1]
        v = color[:, :, :, 2]
  
        u = u.reshape((1, *u.shape))
        v = v.reshape((1, *v.shape))

        loss_img = cv2.merge((u, v))
        loss_img = loss_img[0]

        step_loss, pred = train_step(gray, loss_img)
        step_loss = step_loss.numpy()
        step = optimizer.iterations.numpy()
        lr = optimizer.lr.numpy()

        print(f"Step: {step}, Initial Loss: {step_loss}, epoch{epoch}, lr{lr}")
        if loss_list.is_full:
            if step_loss >= sum(loss_list)/cf.reduce_lr_static_args['patience']:
                optimizer.lr = tf.constant(lr * cf.reduce_lr_static_args['factor'])

        loss_list.append(step_loss)

        assert np.isnan(step_loss) or np.isinf(step_loss), "loss is nan or inf, break"

        if step % cf.pred_step_interval == 0:
            pred = model(gray)
            pred = pred.numpy()

            y = gray[:, :, :, 0]
            img = (cv2.merge((y[0], pred[0])) * 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

            colorold = (color * 255).astype("uint8")[0]
            colorold = cv2.cvtColor(colorold, cv2.COLOR_YUV2BGR)

            grayold = (gray * 255).astype("uint8")[0]

            img = np.hstack([img, grayold, colorold])

            cv2.imwrite(cf.pred_image_path+f"{step}.jpg", img)
        
        if temp_epoch != epoch:
            temp_epoch = epoch
            model.save_weights(cf.model_checkpoint_path, save_format='h5')


if __name__ == '__main__':
    main()
