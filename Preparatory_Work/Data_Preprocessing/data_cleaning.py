import cv2 
import os


"""
我们发现也许下载的图片(爬取的图片）
并非全部都如我们所预料
所以特别提供了这个文件以用来清洗

提供一下函数：
1. manual_check_image 手动展现文件夹下所有图片并选择删除还是跳过
2. auto_check_image 自动筛选将单调色占比大的图片移动到指定目录
3. auto_check_image_from_dict 可以自主提供方法来筛选图片并移动到指定文件夹

"""


def log_print(*args, **kwargs):
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


def get_img_name_generator(root_path: str, format_tuple: tuple=('.jpg', '.png')):
    """
    通过根目录得到要求格式的图片名
    :param root_path: 存放图片的根目录
    :param format_tuple:  支持的格式
    :return:  生成器（迭代器） 其类型是str
    e.g: return  "test.jpg"  其中"test.jpg"存放在root_path下
    """

    for _img_name in os.listdir(root_path):

        if _img_name.endswith(format_tuple):

            yield _img_name


def check_image(*args, check_function):
    """
    检查图片方法简易工厂
    扩展请按照模板实现
    :param args: 位置参数，将完全解包并传递
    :param check_function: 对应方法的实现函数
    :return: 应当满足bool类型的返回
    但如果你实现的功能你非常的清楚和明白，那么
    任意返回都可，诺如此做，请自适应修改下方调用处内容
    """
    return check_function(*args)


'''
模板
[x]         x名字可改变
[x...]      可以有任意x1、x2...
[type: xxx] 声明此处应当为xxx类型

def [check_image_xxx]([img], [arg...], check_function):
    pass..
    
    :return result[type: bool]
# 其中请注意可以有任意位置参数，但关键字参数并不在此列
# 其中img名可改变，但他是必须的
'''


def check_image_is_none(image):
    """
    用于检测图片是否读取成功（是否为空）
    :param image: cv mat/ numpy.array 图片
    :return:bool
    """
    return image is None


def check_image_is_colorful(image, threshold: float=0.7):
    """
    检测图片颜色是否丰富
    思路: 图片中单调色(白到黑,3通道值一致的像素点)占
    图片是否超过阈值比重
    值得一提的是numpy的广播我用的并不熟练
    但他现在比原来快多了
    :param image: cv mat/ numpy.array 图片
    :param threshold: 阈值/单调色占比重
    :return:bool
    """

    total = image.shape[0] * image.shape[1]

    # 最后选择使用了numpy的广播机制
    # 时间下降到0.001s一张图
    image = image.reshape(-1, 3)
    image = image[image[:, 0] == image[:, 1]]
    image = image[image[:, 1] == image[:, 2]]

    return len(image) / total >= threshold

    # total = 0
    # blank_pixel = 0
    # 一张图片0.2s
    # for h in image:
    #     for pixel in h:
    #         total += 1
    #         if pixel[0] == pixel[1] == pixel[2]:
    #             blank_pixel += 1

    # # 第二版
    # blank_pixel = 0
    # total = image.shape[0] * image.shape[1]
    # image = image.reshape(-1, 3)
    # for pixel in image:
    #     # total += 1  # 更改后直接通过hw计算像素点总数
    #     if pixel[0] == pixel[1] == pixel[2]:
    #         blank_pixel += 1
    #
    # print(blank_pixel)


def manual_check_image(root_path: str, format_tuple: tuple=('.jpg', '.png'),
                       wait_time: int=0, remove_key: str="q"):
    """
    手动检查每一个图片
    这个函数我并不建议直接对大批量图片进行操作，会很累
    建议使用auto_check_image后对移出的图片进行精细复检
    或者在使用auto_check_image时选择using_review参数为True
    也可以使用auto_check_image_from_dict来自定义直接的任务
    :param root_path: 存放图片的根目录
    :param format_tuple: 支持的格式
    :param wait_time: （不建议修改）图片的等待时间，为0意味着阻塞
    :param remove_key: 当按下此键时，删除图片，同时这个参数会影响窗口名
    remove_key应当是一个chr类型但python并没有显式表明此类型，请遵守
    :return: None
    """

    for _img_name in get_img_name_generator(root_path, format_tuple):
        _img_path = root_path + '/' + _img_name

        img = cv2.imread(_img_path)

        # 这里不使用断言的原因是我们不希望终止程序
        if check_image(img, check_function=check_image_is_none):
            log_print(_img_path, "no found")
            continue

        window_name = "using {0} to remove".format(remove_key)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(wait_time)

        if key == ord(remove_key):
            os.remove(_img_path)
            log_print("input: {0} and remove: {1}".format(chr(key), _img_path))
        else:
            log_print("input: {0} and continue".format(chr(key)))
            continue

        cv2.destroyAllWindows()

    log_print("一帆风顺", end="\n\n")


def auto_check_image(root_path: str, move_path: str, format_tuple: tuple = ('.jpg', '.png'),
                     threshold: float = 0.7, using_review: bool=False):
    """
    自动筛选所有来自根目录符合条件的图片
    他会提示无法读取的图片并且将不符合要求的图片
    暂时移出root_path到move_path
    使用using_review有助于你最后一次复检，但这需要手工
    :param root_path: 存放图片的根目录
    :param move_path: 移出图片的根目录
    :param format_tuple: 支持的格式
    :param threshold: 阈值，check_image_is_colorful方法会用到
    详情请见check_image_is_colorful，其作用主要是检测图片颜色是否丰富
    :param using_review: 是否使用手工复检，他会调用manual_check_image
    :return: None
    """

    for _img_name in get_img_name_generator(root_path, format_tuple):
        _img_path = root_path + '/' + _img_name

        img = cv2.imread(_img_path)

        if check_image(img, check_function=check_image_is_none):
            log_print(_img_path, "no found")
            continue

        if check_image(img, threshold, check_function=check_image_is_colorful):

            os.renames(_img_path, move_path + '/' + _img_name)
            log_print("moving :", _img_name)

        else:

            log_print("continue: {0}".format(_img_path))

    if using_review:
        manual_check_image(move_path, format_tuple)

    log_print("一帆风顺", end="\n\n")


def auto_check_image_from_dict(root_path: str, move_path: str,
                               format_tuple: tuple = ('.jpg', '.png'),
                               *, check_function_dict: dict):
    """
    通过传递的function字典自动检查图片
    并将不符合要求的图片移动到move_path中
    :param root_path:存放图片的根目录
    :param move_path:移出图片的根目录
    :param format_tuple:支持的格式
    :param check_function_dict:一个符合要求格式的function字典
    其要求如下
        1. function字典中的的function应当是存在且符合check_image模板的
        2. function对应的values中并不需要传递图片,程序会默认传递
        如果你是按照模板来定义的function那么无需担心前2点
        3. 提供额外的参数隐式传递并以元祖形式提供
    e.g:
        auto_check_image_from_dict('mypic', "train", check_function_dict={
            check_image_is_none: (),
            check_image_is_colorful: (0.7,)
        })
        其中check_image_is_none第一个参数应当是image，由程序提供所以传递空元祖
        而check_image_is_colorful除了image还有threshold参数
        我希望是0.7所以传递(0.7,)元祖

        非常建议check_image_is_none应当使用并且放在最前
        否则无法保证image是存在的
    :return:
    """

    for _img_name in get_img_name_generator(root_path, format_tuple):
        _img_path = root_path + '/' + _img_name

        img = cv2.imread(_img_path)
        log_print("image_path: {} checking".format(_img_path))

        for check_function in check_function_dict.keys():

            log_print("runing: {0}, args is {1}".format(check_function.__name__,
                                                        check_function_dict.get(check_function)))

            if check_image(img, *check_function_dict.get(check_function), check_function=check_function):
                os.renames(_img_path, move_path + '/' + _img_name)
                log_print("moving :{0}, because: {1}".format(_img_name, check_function.__name__))
                break

    log_print("一帆风顺", end="\n\n")


if __name__ == '__main__':
    # 手动检查此目录下的所有图片，默认按q键删除，任意其他键跳过
    # manual_check_image('mypic')

    # 自动筛选单调的图片并警告无法读取的图片
    # auto_check_image('mypic', "train")

    # 自动检查mypic目录下所有图片，并将不符合条件的图片移到train
    auto_check_image_from_dict('mypic', "train", check_function_dict={
        check_image_is_none: (),
        check_image_is_colorful: (0.7,)
    })
