from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt
# from io import BytesIO
from io import BufferedReader, BufferedWriter, BytesIO
from .model import ChangedUNetG
from .dataread import img_padding, img_resize, img_check_gray

img_resize_size = 512
model = ChangedUNetG()
model.build((1, img_resize_size, img_resize_size, 3))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model.load_weights(os.path.join(BASE_DIR, 'comic_coloring_app/model_list/model_3v181pro.h5'))#, by_name=True)


def test(request):

    if request.method == 'POST':
        # error_msg = {'errcode': '0006', 'errmsg': '接口特殊不存在（也许是接口被暂时关闭或者并未开放）'}
        try:
            image = request.FILES.get("image")
            # print(request.FILES)
            if not image:
                return HttpResponse("没有检测到数据")
            """
            img_data = plt.imread(BytesIO(imgbyte), "jpg")
            print(type(img_data))
            print(img_data.shape)
            """

            """
            imgbyte = b''
            for c in item.chunks():
                imgbyte += c
                
            # imgbyte = list(map(int, imgbyte))
            # print(np.array(imgbyte, dtype=np.uint8).reshape())
            """

            """
            imgbyte = b''
            for c in item.chunks():
                imgbyte += c
                
            imgarr = np.frombuffer(imgbyte, dtype=np.uint8)
            img_decode = cv2.imdecode(imgarr, 1)
            # imgarr.shape = (580, 822, 3)
            print(img_decode.shape)   
            """

            img_buff = BufferedReader(image)

            img_byte = BufferedReader.read(img_buff)

            nparr = np.frombuffer(img_byte, dtype=np.uint8)

            img_decode = cv2.imdecode(nparr, 1)
            if img_decode is None:
                return HttpResponse("图片损坏或不符合格式")

            img_decode = img_padding()(img_decode)
            img_decode = img_resize((img_resize_size, img_resize_size))(img_decode)
            if not img_check_gray(img_decode):
                # img_decode = cv2.cvtColor(img_decode, cv2.COLOR_RGB2YUV)
                img_decode = cv2.cvtColor(img_decode, cv2.COLOR_RGB2GRAY)
                img_decode = cv2.merge((img_decode, img_decode, img_decode))

            img_decode = img_decode.reshape((1, *img_decode.shape))

            img_decode = img_decode.astype('float32') / 255.0
            # print(img_decode.shape)

            pred = model(img_decode)
            pred = pred.numpy()

            y = img_decode[:, :, :, 0]
            img = (cv2.merge((y[0], pred[0])) * 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

            success, encoded_image = cv2.imencode(".jpg", img)

            img_bytes = encoded_image.tobytes()
            # bt = BytesIO()
            # bt.write(img_bytes)
            #
            # img_byte = BufferedWriter(bt)
            # img_buff = BufferedWriter.write(img_byte)

            # response = JsonResponse(
            #     error_msg,
            #     content_type='application/json;charset=utf-8',
            #     json_dumps_params={"ensure_ascii": False}
            # )

            response = HttpResponse(
                content=img_bytes,
                content_type='image/jpeg'
            )

            return response

        except Exception as err:
            print(err)
            return HttpResponse("遇到异常,多次尝试无果请联系")

    return HttpResponse("错误的访问方式")


def error_404(request, exception):
    """
    访问不存在(并未使用)
    :param request:
    :param exception:
    :return: 提示信息
    """
    return HttpResponseRedirect("/")
