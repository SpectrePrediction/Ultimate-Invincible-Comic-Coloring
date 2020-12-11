# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import wraps
import sys
import cv2
import numpy as np


def log_print(*args, **kwargs):
    """
    注: 在cmd等不支持彩色的命令框 log_print函数无法发挥他颜色的效果
    :param args: 一帆风顺
    :param kwargs:  一帆风顺
    :return:
    """
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


def data_read_function(debug: bool=False):
    def decorator(func):
        def _check_func(*args, **kwargs):
            assert func.__code__.co_argcount - 1 >= 0, """
                被data_read_function修饰的函数应当有1个以上参数
            """

            assert len(args)+len(kwargs)+1 == func.__code__.co_argcount, f"""
                输入应当是{func.__code__.co_argcount - 1}个参数，但是输入了{len(args)+len(kwargs)}个
                args:{str(args)}, kwargs:{str(kwargs)}
                可能是: 
                    被data_read_function修饰且debug为False的函数在第一次传参时仅需要传
                除去第一个位置参数外的其他参数，并且形式不限,请注意是否额外传入了第一个参数
                (因为按照模板，第一个参数应当是image(变量名可改),此参数将在参与运算时自动call)
                e.g: def test(image, size) using: test(1)(2)  则size = 1 , image = 2
                请不要如此操作test(1,3)(2),这将引发此类错误
            """

            assert not kwargs.get(func.__code__.co_varnames[0]), f"""
                    第一个参数 {func.__code__.co_varnames[0]} 将会在参与运算时自动call
                所以你不应当显式的指定他的值
                e.g: def test(image, size) using: test(1)(2)  则size = 1 , image = 2
                请不要如此操作test(image=1)(2),这将引发此类错误
            """

            def _temp_func(*args, **kwargs):
                def _call_func(img):
                    return func(img, *args, **kwargs)

                return wraps(func)(_call_func)
            return _temp_func(*args, **kwargs)

        if debug is True:
            # 此处is True 并不是多余的
            # 是debug模式，直接返回原函数，不做修改
            return func

        # 非debug时,修饰检查并构造我们想要的函数
        return _check_func

    # if isinstance(debug, types.FunctionType):
    if callable(debug):
        # 此时@data_read_function没有( )被调用
        # debug 是 被修饰函数
        return decorator(debug)

    # 此时@data_read_function有( )被调用
    # debug 按照提示应当是bool
    return decorator


"""
data_read_function修饰器介绍:
被data_read_function修饰器修饰会有如下效果
    1. 被data_read_function修饰的函数应当有1个以上参数
    2. 第一个参数应当是用来接受图像
    3. 在使用被修饰函数时，不需要提供第一个参数，第一个
参数将会在使用此方法的时候自动传入

data_read_function修饰时
可以使用@data_read_function或者@data_read_function()

后者可以传入一个参数:是否是debug模式,默认是False
@data_read_function(debug=True)
@data_read_function(True)

使用debug模式data_read_function修饰器将失效
函数的使用方法回归平常，建议定义全局变量控制

e.g:
    @data_read_function
    def test(img, r):
        print(f"img{img}, r{r}")
        return 1
    
print(test(2)(1))
或者print(test(r=2)(1))
输出：
    >>> img1, r2
    >>> 1

如果你不想用修饰器，那么可以使用闭包
但请同样遵守第一个参数为图像接收参数
e.g:
    def img_resize(size):
        def resize(img):
            print(img, size)
            return 1
        return wraps(img_resize)(resize)
    
print(img_resize(2)(1))
效果和上面相似

"""


class BasicDataReader(object):

    def __init__(self):
        pass

    def read_data(self, function_list):
        """
        我会在未来重新
        :param function_list:
        :return:
        """
        raise RuntimeError("read_data not define")

    def read_all_data(self, function_list):
        """
        我会在未来重新
        :param function_list:
        :return:
        """
        raise RuntimeError("read_all_data not define")

    @staticmethod
    def show_progress(total: int) -> str:
        """
        进度条，迭代器每一次迭代前进 1/total
        e.g:它来自以前的代码
        :param total: 总数
        :return: 进度条字符串生成器
        """
        for num in range(1, total + 1):
            print_num = num * (50 / total)
            print_str = '[%-50s] %s/%s ' % ('=' * int(print_num), str(num), str(total))
            yield print_str


class CacheList(list):

    def __init__(self, data_cache: int=0):
        super(CacheList, self).__init__()
        self.data_cache = data_cache if data_cache > 0 else 0

    @property
    def is_full(self):
        return self.__len__() >= self.data_cache and self.data_cache

    def append(self, data: object):
        if self.__len__() >= self.data_cache and self.data_cache:
            self.pop(0)

        super(CacheList, self).append(data)


class DataReader(BasicDataReader):

    def __init__(self, image_txt_path: str, function_list: list,
                 batch_size: int=16, *, is_show_progress: bool=True,
                 read_data_cache: int=0, is_completion: bool=True):
        """

        :param image_txt_path:
        :param function_list:
        :param batch_size:
        :param is_show_progress:
        :param read_data_cache: 为零意味着不设定缓冲，将全部数据一次读取
        注： 建议为批大小的倍数, 图片所占内存并非直接由缓冲决定
        而且由缓冲被批大小补全决定
        :param is_completion: 是否补全文件数为批大小的倍数
        注：此选项为True会改变文件数目,并且多余补偿部分是来自文件本身
        """

        super(DataReader, self).__init__()

        self.function_list = function_list
        self.is_show_progress = is_show_progress
        self.is_completion = is_completion
        self.batch_size = int(batch_size) if batch_size > 0 else 1
        self.ont_epoch_step = 1
        self.step = 0
        self.epoch = 0

        with open(image_txt_path) as f_read:
            self._txt_content = f_read.readlines()

        self._txt_content = [path.split() for path in self._txt_content]

        assert self._txt_content.__len__() >= self.batch_size,\
            f"""
                    批大小应当小于等于数据总量
                    请检查TXT中是否总量达不到批大小
                    TXT中读取到的总量是：{self._txt_content.__len__()}
                    批大小为：{self.batch_size}
             """
        temp = self._txt_content.__len__() % self.batch_size
        if self.is_completion and temp != 0:
            self._txt_content.extend(self._txt_content[: self.batch_size - temp])
            log_print(f"is_completion为{self.is_completion},文件数量补全为{self._txt_content.__len__()}")

        self.total = self._txt_content.__len__()

        self.read_data_cache = read_data_cache if read_data_cache else self.total

        assert self.read_data_cache >= self.batch_size, f"""
            缓冲区大小：{self.read_data_cache} 应当大于等于批大小：{self.batch_size}
        """

        if self.is_completion and self.read_data_cache % self.batch_size != 0:
            self.read_data_cache += self.batch_size - self.read_data_cache % self.batch_size
            log_print(f"is_completion为{self.is_completion},缓冲区补全为{self.read_data_cache}")

        assert self._txt_content.__len__() >= self.read_data_cache, \
            f"""
                    缓冲区大小同样应当小于等于总量
                    请检查TXT中是否总量达不到批大小
                    TXT中读取到的总量是：{self._txt_content.__len__()}
                    缓冲区大小：{self.read_data_cache}
                    注意：当较低数量的文件总数，批大小不能整除
                    触发补全机制时，缓冲区是暂时允许超过 原 文件总数的
             """
        self.total_step = self.total // self.batch_size
        self.data_iter = self.get_data_iter(function_list=self.function_list)

    def get_data_iter(self, function_list):
        """
        很抱歉我用了这样简单粗暴且拉跨的写法
        我需要时间来整理逻辑，但比较需要先用起来
        所以我保证将会在未来版本将此处重写

        :param function_list: 需要执行的函数列表
        其应当遵守:
            使用data_read_function修饰器或者闭包
            具体请参加data_read_function详解
        :return:
        """

        if self.read_data_cache == self.total:
            color_data = []
            gray_data = []
            progress = self.show_progress(len(self._txt_content))

            for color_img, gray_img in self._txt_content:

                bgr_img = cv2.imread(color_img)
                onechannel_img = cv2.imread(gray_img)

                for func in function_list:

                    bgr_img = func(bgr_img)
                    onechannel_img = func(onechannel_img)

                if self.is_show_progress:
                    sys.stdout.flush()
                    sys.stdout.write('\r' + next(progress))

                color_data.append(bgr_img)
                gray_data.append(onechannel_img)

            while True:
                yield np.array(color_data), np.array(gray_data)

        else:
            while True:
                color_data = CacheList(self.read_data_cache)
                gray_data = CacheList(self.read_data_cache)
                progress = self.show_progress(self._txt_content.__len__())
                step = 0

                for color_img, gray_img in self._txt_content:

                    bgr_img = cv2.imread(color_img)
                    onechannel_img = cv2.imread(gray_img)

                    for func in function_list:
                        bgr_img = func(bgr_img)
                        onechannel_img = func(onechannel_img)

                    if self.is_show_progress:
                        sys.stdout.flush()
                        sys.stdout.write('\r' + next(progress))

                    color_data.append(bgr_img)
                    gray_data.append(onechannel_img)
                    step += 1

                    if step >= self.read_data_cache:
                        yield np.array(color_data), np.array(gray_data)
                        step = 0

    def __getitem__(self, item):
        _temp_step = self.step
        if _temp_step == 0:
            self.color_data, self.gray_data = self.data_iter.__next__()

        if self.read_data_cache == self.total:
            total_step = self.total_step
            if _temp_step > total_step - 1:
                _temp_step = self.step = 0
                # self.epoch += 1
        else:
            total_step = self.read_data_cache // self.batch_size
            if _temp_step > total_step - 1:
                _temp_step = self.step = 0
                self.color_data, self.gray_data = self.data_iter.__next__()

        out_color_data = self.color_data[self.batch_size * _temp_step: self.batch_size * (_temp_step + 1)]
        out_gray_data = self.gray_data[self.batch_size * _temp_step: self.batch_size * (_temp_step + 1)]

        self.step += 1
        if self.ont_epoch_step - 1 >= self._txt_content.__len__() // self.batch_size:
            self.ont_epoch_step = 1
            self.epoch += 1

        self.ont_epoch_step += 1

        return self.epoch, out_color_data, out_gray_data


@data_read_function
def img_padding(image):

    width, high, channel = image.shape if len(image.shape) == 3 else (*image.shape, 1)

    dim_diff = np.abs(high - width)

    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if high <= width else (pad1, pad2, 0, 0)
    top, bottom, left, right = pad

    img_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    return img_pad


@data_read_function
def img_resize(image, resize):
    re_img = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    return re_img


@data_read_function
def gray_img_reshape(image):
    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    return image


@data_read_function
def img_BGR2YUV(image):
    if not img_check_gray(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    return image


def img_check_gray(image):
    total = image.shape[0] * image.shape[1]

    # 最后选择使用了numpy的广播机制
    # 时间下降到0.001s一张图
    image = image.reshape(-1, 3)
    image = image[image[:, 0] == image[:, 1]]
    image = image[image[:, 1] == image[:, 2]]

    return len(image) / total >= 0.9


if __name__ == '__main__':

    dr = DataReader(r"D:\mhss\code\train.txt", [
            img_padding(),
            img_resize((512, 512)),
            # gray_img_reshape()
        ],
        read_data_cache=20,
        batch_size=20,
        is_completion=True
    )

    import time
    for epoch, color, gray in dr:
        color = color.astype("float32") / 255.0
        gray = gray.astype("float32") / 255.0
        time.sleep(1)
        cv2.imshow("color", color[0])
        cv2.imshow("gray", gray[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(color.shape, gray.shape, epoch)

    # train_data = ImageDataGenerator()
    # train_generator = train_data.flow_from_directory(r".\mypic")
