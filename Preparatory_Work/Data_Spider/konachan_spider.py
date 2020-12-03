import requests
from bs4 import BeautifulSoup
import sys
import os


picture_header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
picture_root_url = 'https://konachan.net/post?page='
# konachan虽然可以访问，但速度无法恭维
"""
# 若是f { } 这里报错，请修改使用其他方法即可 e.g: .format 等等
# 旧版本python 也许支持有限 但这不是大问题
"""


def log_print(*args, **kwargs):
    """
    注: 在cmd等不支持彩色的命令框 log_print函数无法发挥他颜色的效果
    :param args: 一帆风顺
    :param kwargs:  一帆风顺
    :return:
    """
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


class PictureDownloader:

    def __init__(self, star_page: int=0, num_page: int=1, url: str=picture_root_url,
                 time_out: int=None, header: dict=picture_header, is_show_progress: bool=True):
        """
        PictureDownloader图片下载者
        通过指定的页数得到所有图片的下载地址
        然后使用download_to函数将其下载到一个文件夹中

        :param star_page:   第几页开始 默认从头开始
        :param num_page:    下载几页 默认下载一页
        注： 包含此页在内，例如第一页开始，num_page为1，则下载第一页结束
        :param url:         下载图片的根目录地址
        :param time_out:    访问超时时间设置，为None则不限
        注:  更新，现在超过时间访问将会跳过，而不是直接抛出异常
        :param header:      访问头，使用默认即可
        :param is_show_progress:  是否展示进度条
        注: 在cmd等不支持彩色的命令框 log_print函数无法发挥他颜色的效果
        """

        self.star_page = star_page if star_page >= 0 else 0
        self.end_page = star_page + num_page

        self.url = url
        self.time_out = time_out
        self.header = header

        self.is_show_progress = is_show_progress
        self.picture_url_list = self.get_picture_url_list()

    def __del__(self):
        log_print("PictureDownloader 已退出")

    def get_picture_url_list(self) -> list:
        """
        get_picture_url_list得到图片下载地址的列表
        他通过构造函数自动执行并获取，同样是后面下载的必需品
        :return: 图片下载地址的列表
        """

        progress = self.show_progress(self.end_page - self.star_page)

        log_print(f"正在获取图片下载地址 from {self.url}")
        temp_list = list()

        for page in range(self.star_page, self.end_page):

            temp_url = self.url + str(page)

            try:

                html = requests.get(temp_url, headers=self.header, timeout=self.time_out).text
                soup = BeautifulSoup(html, 'html.parser')

                for img in soup.find_all('img', class_="preview"):
                    target_url = img['src']
                    temp_list.append(target_url)

            # 若是f {page} 这里报错，请修改使用其他方法即可 e.g: .format 等等
            except requests.exceptions.ReadTimeout:
                log_print(f"ReadTimeout countinue: 跳过第{page} 页")

            except requests.exceptions.ConnectionError:
                log_print(f"Connectiontimed out countinue: 跳过第{page} 页")

            if self.is_show_progress:
                sys.stdout.flush()
                sys.stdout.write('\r' + next(progress))
            else:
                log_print(f"第{page}页获取完, 还有{self.end_page - page - 1}页")

        sys.stdout.write('\n')
        log_print("一帆风顺， 准备下载")

        return temp_list

    @staticmethod
    def save_picture(picture_data: bytes, save_path: str):
        """
        保存图片（注：早期中判断已被移出到其他地方，现在save_picture只负责保存）
        :param picture_data: 图片数据
        :param save_path: 保存地址
        :return: None
        """
        with open(save_path, 'wb') as picture:
            picture.write(picture_data)
    
    def download_to(self, save_root_path: str, is_cover: bool=False):
        """
        下载到指定文件夹
        :param save_root_path: 保存图片的根目录（文件夹）
        注： 可以不带‘/’，例如"."与"./"是一致的
        :param is_cover: 如果图像存在，是否覆盖 默认否
        :return: None
        """

        if not save_root_path.endswith('/'):
            save_root_path += '/'

        progress = self.show_progress(len(self.picture_url_list))

        for picture_url in self.picture_url_list:

            picture_name = picture_url.split("/")[-1]
            save_path = save_root_path + picture_name

            if os.path.exists(save_path) and not is_cover:
                log_print(f"{picture_name}文件已存在且is_cover是False:跳过保存")
                continue

            try:

                picture_data = requests.get(picture_url, timeout=self.time_out).content
                self.save_picture(picture_data, save_path)

            except requests.exceptions.ReadTimeout:
                log_print(f"ReadTimeout countinue: 跳过{picture_name}图片")

            except requests.exceptions.ConnectionError:
                log_print(f"Connectiontimed out countinue: 跳过{picture_name}图片")

            if self.is_show_progress:
                sys.stdout.flush()
                sys.stdout.write('\r' + next(progress))

        sys.stdout.write('\n')
        log_print("一帆风顺, 下载结束")

    @staticmethod
    def show_progress(total: int)->str:
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


pd = PictureDownloader(star_page=0, num_page=20, is_show_progress=True, time_out=1)
pd.download_to("./downimg")  # 应当有此目录，程序并不会自动创建
