from img_to_sketch import get_sketch_from_path
import numpy as np
import random
import os

'''
提供
1.根目录修改所有图片文件名                          rename_all_image_from_root
2.读取根目录所有图片并转化成线稿并放入指定文件夹    sketch_all_image_from_root
3.通过两个根目录生成列表并划分训练集和测试集        get_image_list_from_root
4.等于2和3连用，将图片处理再生成对应训练集和测试集  sketch_all_image_and_get_list

# 使用方法
通过root根目录
1. 可选！使用rename_all_image_from_root修改名字
2. 然后使用sketch_all_image_from_root将此目录下所有的图片进行线稿处理并放入指定文件夹
3. 最后通过get_image_list_from_root生成对应的train.txt和test.txt

注意：如果2和3没有特殊情况，可以直接使用sketch_all_image_and_get_list

e.g:
其中"mypic"文件夹是我下载下来的图片
"train"文件夹是我用来保存生成草图的文件夹

根目录修改所有图片文件名
rename_all_image_from_root("mypic", name_generator=number_name_generator())
读取根目录所有图片并转化成线稿并放入指定文件夹
sketch_all_image_from_root("mypic", "train", using_mode_dict={"1": 0.9, "2": 0.1})
通过两个根目录生成列表并划分训练集和测试集
get_image_list_from_root("mypic", "train")

或者：使用sketch_all_image_and_get_list以取代
sketch_all_image_from_root和get_image_list_from_root
用法： sketch_all_image_and_get_list("mypic", "train", using_mode_dict={"1": 0.9, "2": 0.1})

运行结果应当为
1.倘若选择了修改名字mypic下所有的图片应当被改名
2.mypic下图片生成对应的草图到train
3.在此文件同目录下生成train.txt和test.txt

'''


def log_print(*args, **kwargs):
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


def number_name_generator(star: int=0):
    """
    简易的数字名字生成器
    :param star: 开始值
    :return: str类型的数字
    """
    while True:
        yield str(star)
        star += 1


def rename_all_image_from_root(root_path: str, format_tuple: tuple=('.png', '.jpg'),
                               *, name_generator, rename_same_format: str=None):
    """
    对根目录下所有符合结尾要求的文件进行重命名
    :param root_path: 存放图像的文件夹名
    :param format_tuple: 符合要求的文件名后缀结尾,不符合要求的文件不会被处理
    :param name_generator: 需要显式传参:名字生成器, 可以使用number_name_generator
    e.g:自定义名字生成器迭代的内容应当是str类型
    :param rename_same_format: 需要显式传参:是否更改为统一后缀名, 为None则维持原样
    :return: None
    """

    if rename_same_format:
        assert rename_same_format.startswith('.'), "格式符正确格式应当为'.xxx'而非'xxx'"

    for _img_name in os.listdir(root_path):

        if _img_name.endswith(format_tuple):
            _img_format = rename_same_format if rename_same_format else '.' + _img_name.split('.')[-1]
            _old_name = root_path + '/' + _img_name
            _new_name = root_path + '/' + name_generator.__next__() + _img_format
            log_print("正在处理: {0} 更名为 {1}".format(_old_name, _new_name))
            os.renames(_old_name, _new_name)

    log_print("一帆风顺", end="\n\n")


def get_image_list_from_root(root_path: str, sketch_root_path: str,
                             format_tuple: tuple = ('.png', '.jpg'),
                             train_txt_path: str = "train.txt",
                             test_txt_path: str = "test.txt",
                             test_split: float = 0.1):
    """
    通过两个根目录生成列表并划分训练集和测试集
    :param root_path: 普通图片的存放地址
    :param sketch_root_path: 线图的存放地址
    :param format_tuple: 符合要求的文件名后缀结尾,不符合要求的文件不会被处理
    :param train_txt_path: train.txt的存放路径，当然你也可以修改他的名字
    :param test_txt_path: test.txt的存放路径，当然你也可以修改他的名字x2
    :param test_split:测试集所占比例，他应当是一个浮点数，请注意
    :return:
    """

    temp_list = list()

    with open(train_txt_path, "w") as ftrain_writer:
        with open(test_txt_path, "w") as ftest_writer:
            for _img_name in os.listdir(root_path):

                if _img_name.endswith(format_tuple):

                    _old_name = root_path + '/' + _img_name
                    _new_name = sketch_root_path + '/' + _img_name

                    if not os.path.exists(_new_name):
                        log_print("警告！", _old_name, "没有找到对应处理后的图片, 请检查")
                        continue

                    temp_list.append(_old_name + ' ' + _new_name + '\n')

            total = len(temp_list)
            split = total - int(total*test_split)
            log_print("已获取全部图片名")
            log_print("图片总量为:{0}, 测试集比例:{1}".format(total, test_split))

            train_list, test_list = temp_list[:split+1], temp_list[split:]
            log_print("train_list量为:{0}, test_list量为:{1}".format(len(train_list), len(test_list)))

            log_print("开始写入train_list")
            for train_path in train_list:
                ftrain_writer.write(train_path)
            log_print("train_list完成")

            log_print("开始写入test_list")
            for test_path in test_list:
                ftest_writer.write(test_path)
            log_print("test_list完成")

    log_print("一帆风顺", end="\n\n")


def sketch_all_image_from_root(root_path: str, out_path: str,
                               format_tuple: tuple = ('.png', '.jpg'),
                               *, using_mode_dict: dict):
    """
    将此root_path目录下所有的图片进行线稿处理并放入指定out_path文件夹
    :param root_path: 存放要处理图片的根目录
    :param out_path:  修改后存放的目录，默认名字不变，对应原图
    :param format_tuple:  需要修改图像的后缀
    :param using_mode_dict:  使用修改线稿的模式概率字典，模式详情请看img_to_sketch文件
    using_mode_dict例如{"1": 0.9, "2": 0.1}，意味着其中“1”方法出现的概率占90%
    请注意如下：
        1.所有的概率和应当为1
        2.所有的dict的key也就是模式应当存在
        3.请注意这是出现的概率，并非结果一定是期望的分布，所以尽量悲观去设计
        例如只希望出现几张“2”模式下处理的图，那么我的“2”概率可以设定为0.05甚至更小
    :return:
    """

    assert root_path != out_path, "root_path不应当与out_path相同"
    assert sum(using_mode_dict.values()) == 1.0, "using_mode_dict的概率相加应当为1"

    p = np.array(list(using_mode_dict.values()))

    for _img_name in os.listdir(root_path):

        if _img_name.endswith(format_tuple):

            _old_name = root_path + '/' + _img_name
            _new_name = out_path + '/' + _img_name

            mode = str(np.random.choice(list(using_mode_dict.keys()), p=p.ravel()))
            log_print("正在使用方法{0}处理: {1} 去到 {2}".format(mode, _old_name, _new_name))
            get_sketch_from_path(_old_name, _new_name, mode=mode)

    log_print("一帆风顺", end="\n\n")


def sketch_all_image_and_get_list(root_path: str, sketch_root_path: str,
                                  format_tuple: tuple = ('.png', '.jpg'),
                                  train_txt_path: str = "train.txt",
                                  test_txt_path: str = "test.txt",
                                  test_split: float = 0.1, *, using_mode_dict: dict):

    sketch_all_image_from_root(root_path, sketch_root_path, format_tuple, using_mode_dict=using_mode_dict)
    get_image_list_from_root(root_path, sketch_root_path, format_tuple, train_txt_path, test_txt_path, test_split)

    return None


if __name__ == '__main__':
    # 根目录修改所有图片文件名
    rename_all_image_from_root("mypic", name_generator=number_name_generator())
    # 读取根目录所有图片并转化成线稿并放入指定文件夹
    # sketch_all_image_from_root("mypic", "train", using_mode_dict={"1": 0.9, "2": 0.1})
    # 通过两个根目录生成列表并划分训练集和测试集
    # get_image_list_from_root("mypic", "train")

    # 上面两者的结合
    sketch_all_image_and_get_list("mypic", "train", using_mode_dict={"1": 0.9, "2": 0.1})
