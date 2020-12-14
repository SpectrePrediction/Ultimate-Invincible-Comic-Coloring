import cv2
import tensorflow as tf


def euclidean_dist(x, y):
    """
    矩阵 欧几里得距离
    :param x: tensor
    :param y: tensor
    :param axis: 相加维度
    :return: tensor
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x - y)))
    # return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))


class CosineSimilarityEuclideanLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return 10 * tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))) / \
               tf.abs(-tf.reduce_mean(tf.keras.losses.cosine_similarity(y_pred, y_true, 1)) + 0.001)


class BlurEuclideanDistLoss(tf.keras.losses.Loss):
    """
        img = [batch, H, W, C]
        :param img: 应当是tensor           y_true
        :param inferred_img: 应当是tensor  y_pred
        :param axis: 相加维度
        :return: tensor[batch, H, C]
    注意:测试未通过,无法使用.除非自己用tf重写高斯模糊
    """

    def call(self, img, inferred_img):

        img = img.numpy()
        inferred_img = inferred_img.numpy()

        shape = img.shape
        assert shape == inferred_img.shape, "img shape and inferred_img shape should be same"

        img = img.reshape(shape[0] * shape[2], shape[1], shape[3])
        inferred_img = inferred_img.reshape(shape[0] * shape[2], shape[1], shape[3])

        blur_img3 = cv2.blur(img, (3, 3))  # 重写不难
        blur_img5 = cv2.blur(img, (5, 5))

        inferred_blur_img3 = cv2.blur(inferred_img, (3, 3))
        inferred_blur_img5 = cv2.blur(inferred_img, (5, 5))

        if shape[3] == 1:
            """
            如果是单通道需要在处理后重新指定[batch, H, 1]
            否则[batch, H, 1]-》[batch, H]
            """
            blur_img3 = blur_img3.reshape(shape[0] * shape[2], shape[1], shape[3])
            blur_img5 = blur_img5.reshape(shape[0] * shape[2], shape[1], shape[3])
            inferred_blur_img3 = inferred_blur_img3.reshape(shape[0] * shape[2], shape[1], shape[3])
            inferred_blur_img5 = inferred_blur_img5.reshape(shape[0] * shape[2], shape[1], shape[3])

        out = tf.reduce_sum([
            euclidean_dist(inferred_img, img),
            euclidean_dist(inferred_blur_img3, blur_img3),
            euclidean_dist(inferred_blur_img5, blur_img5),
        ])

        return out


def cosine_similarity(img, inferred_img, axis=1):
    """
    img = [batch, H, W, C]
    :param img: 应当是tensor           y_true
    :param inferred_img: 应当是tensor  y_pred
    :param axis: 相加维度
    :return:tensor [batch, H, C]
    """
    return tf.reduce_mean(tf.keras.losses.cosine_similarity(img, inferred_img, axis))


if __name__ == '__main__':
    input_shape = (8, 512, 512, 2)
    x = tf.random.normal(input_shape)
    y = tf.random.normal(input_shape)

    # 2896
    print(euclidean_dist(x, y))
    # -0.00000188
    print(cosine_similarity(x, y))
    # 4443
    print(BlurEuclideanDistLoss()(x, y))
    # 16821
    print(CosineSimilarityEuclideanLoss()(x, y))
