from django.shortcuts import render
from django.http import HttpResponse
import requests


def define_page(request):
    if request.method == 'GET':

        # return HttpResponse(
        #     "提示：请使用post方法访问任何接口<br>"
        #     "这是测试页面<br>"
        # )

        return render(request, r"post_image.html")

    return HttpResponse("错误的访问方式")
