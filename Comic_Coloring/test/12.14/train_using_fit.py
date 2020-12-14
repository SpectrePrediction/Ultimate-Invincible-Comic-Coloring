import tensorflow as tf
import losses as loss
import datautil as util
import models as md
# import numpy as np
import conf as cf
import cv2


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def main():
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
    lr_metric = get_lr_metric(optimizer)

    model.compile(optimizer=optimizer,
                  loss=loss.CosineSimilarityEuclideanLoss(),
                  metrics=['accuracy', lr_metric])

    for epoch, color, gray in dr:
        gray = gray.astype('float32') / 255.0
        color = color.astype('float32') / 255.0
        u = color[:, :, :, 1]
        v = color[:, :, :, 2]
  
        u = u.reshape((1, *u.shape))
        v = v.reshape((1, *v.shape))

        loss_img = cv2.merge((u, v))
        loss_img = loss_img[0]

    nan_stop = tf.keras.callbacks.TerminateOnNaN()
    model_save = tf.keras.callbacks.ModelCheckpoint(**cf.model_checkpoint_args)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(**cf.reduce_lr_on_plateau_args)

    model.fit(gray, loss_img, **cf.fit_args, callbacks=[reduce_lr, model_save, nan_stop])


if __name__ == '__main__':
    main()
