import requests
from bs4 import BeautifulSoup
import sys
import os


picture_header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
picture_root_url = 'https://konachan.net/post?page='
# konachan虽然可以访问，但速度无法恭维


def log_print(*args, **kwargs):
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


class PictureDownloader:

    def __init__(self, star_page: int=0, num_page: int=1, url: str=picture_root_url,
                 time_out: int=None, header: dict=picture_header, is_show_progress: bool=True):
        """
        PictureDownloader图片下载者
        :param star_page:   第几页开始 默认从头开始
        :param num_page:    下载几页 默认下载一页
        注： 包含此页在内，例如第一页开始，num_page为1，则下载第一页结束
        :param url:         下载图片的根目录地址
        :param time_out:    访问超时时间设置，为None则不限
        :param header:      访问头，使用默认即可
        :param is_show_progress:  是否展示进度条
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

        progress = self.show_progress(self.end_page - self.star_page)

        log_print("正在获取图片下载地址 from {0}".format(self.url))
        temp_list = list()

        for page in range(self.star_page, self.end_page):

            temp_url = self.url + str(page)

            html = requests.get(temp_url, headers=self.header, timeout=self.time_out).text
            soup = BeautifulSoup(html, 'html.parser')

            for img in soup.find_all('img', class_="preview"):
                target_url = img['src']
                temp_list.append(target_url)

            if self.is_show_progress:
                sys.stdout.flush()
                sys.stdout.write('\r' + next(progress))
            else:
                log_print("第{0}页获取完, 还有{1}页".format(page, self.end_page - page - 1))

        sys.stdout.write('\n')
        log_print("一帆风顺， 准备下载")

        return temp_list

    @staticmethod
    def save_picture(picture_data: bytes, save_path: str):
        with open(save_path, 'wb') as picture:
            picture.write(picture_data)
    
    def download_to(self, save_root_path: str, is_cover: bool=False):

        if not save_root_path.endswith('/'):
            save_root_path += '/'

        progress = self.show_progress(len(self.picture_url_list))

        for picture_url in self.picture_url_list:

            picture_name = picture_url.split("/")[-1]
            save_path = save_root_path + picture_name

            if os.path.exists(save_path) and not is_cover:
                log_print("{0}文件已存在且is_cover是False:跳过保存".format(picture_name))
                continue

            picture_data = requests.get(picture_url, timeout=self.time_out).content
            self.save_picture(picture_data, save_path)

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


pd = PictureDownloader(star_page=0, num_page=20, is_show_progress=True)
pd.download_to("./downimg")
