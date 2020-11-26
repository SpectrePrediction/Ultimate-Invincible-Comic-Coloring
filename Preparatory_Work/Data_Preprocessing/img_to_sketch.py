import cv2
import numpy as np


def img_to_sketch(cv_3cannel_img, mode='1'):
    """
    img_to_sketch的总方法
    负责图像的断言检测以及通过mode参数分配方法
    :param cv_3cannel_img: 输入应当是cv读取的3通道图像
    :param mode: 使用的模式，效果各异，可以各种尝试一下后选择
    e.g:我们选择两种都用，但mode1的效果我们更喜欢
    e.g:需要完善更多方法的请按照格式创建函数(如下）
    :return:
    """
    assert len(cv_3cannel_img.shape) == 3, "cannel应当为3,且是bgr"

    try:
        return globals()['img_to_sketch_' + mode](cv_3cannel_img)

    except KeyError as key_err:

        raise KeyError("你似乎使用了一个没有定义的方法："
                       "mode={0},error={1}不存在".format(mode, key_err))


'''
模板
[x]         x名字可改变
[type: xxx] 声明此处应当为xxx类型

def img_to_sketch_[mode]([cv_3cannel_img]):
    pass..
    
    :return cv_img[type: np.array]

'''


def img_to_sketch_1(cv_3cannel_img):

    img_gray = cv2.cvtColor(cv_3cannel_img, cv2.COLOR_BGR2GRAY)

    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)

    img_sketch = cv2.divide(img_gray, 255 - img_blur, scale=256)

    return img_sketch


def img_to_sketch_2(cv_3cannel_img):

    img_gray = cv2.cvtColor(cv_3cannel_img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0, sigmaY=0)

    img_sketch = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=7, C=2)

    return img_sketch


def get_sketch_from_path(input_img_path: str, out_img_path: str, *, mode: str):
    """
    通过路径读取图片并保存为草图线稿
    :param input_img_path: 输入图片路径
    :param out_img_path: 输出图片路径
    :param mode: 模式,应当为str 但其内容是'1','2'等，请保证有其实现的方法
    其中mode="1",对应img_to_sketch_1方法
    :return: None
    """
    img = cv2.imread(input_img_path)
    assert img is not None, 'Image 没找到或者检查路径是否有中文 '

    img_sketch = img_to_sketch(img, mode=mode)
    cv2.imwrite(out_img_path, img_sketch)


if __name__ == '__main__':
    # 直接通过读取图片并保存为草图线稿
    get_sketch_from_path(r'test.jpg', r'test2.jpg', mode="2")

    # 通过传入图片，得到输出，并保存
    # cv2.imwrite(r'test1.jpg', img_to_sketch(cv2.imread(r'test.jpg'), mode="1"))
    # 如果你有更高效的读取方法，可通过后者实现
