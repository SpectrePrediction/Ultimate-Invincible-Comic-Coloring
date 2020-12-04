# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import wraps


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
        raise RuntimeError("__init__ not define")

    def read_data(self):
        raise RuntimeError("read_data not define")


if __name__ == '__main__':
    DEBUG = False

    @data_read_function(DEBUG)
    def img_resize2(img, size):
        print(img, size)
        return 1

    def img_resize(size):
        def resize(img):
            print(img, size)
            return 1
        return wraps(img_resize)(resize)

    # 效果一致
    DataReader([
        img_resize("size"),
        img_resize2("size")
    ])

    # train_data = ImageDataGenerator()
    # train_generator = train_data.flow_from_directory(r".\mypic")
