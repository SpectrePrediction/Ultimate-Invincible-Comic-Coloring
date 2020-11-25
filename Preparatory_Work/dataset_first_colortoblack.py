import os

from PIL import Image


original_images_path = r'.\data\OriginalImages'#原始彩色图像路径
color_images_path = r'.\data\ColorImages'#处理后彩色图像路径
grayscale_images_path = r'.\data\GrayscaleImages'#处理后图像路径
combined_images_path = r'.\data\CombinedImages'#拼接图像路径

resize_height = 256
resize_weidth = 256

def find_images(path):#出于不同文件后缀考虑 寻找 后缀为jpg png的文件
    result = []
    for filename in os.listdir(path):
        _, ext = os.path.splitext(filename.lower())
        if ext == ".jpg" or ext == ".png":
            result.append(os.path.join(path, filename))
            pass
        pass
    result.sort()
    return result

if __name__ == '__main__':
    search_result = find_images(original_images_path)#寻找彩色图像路径
    for image_path in search_result:
        img_name = image_path[len(original_images_path):]
        img = Image.open(image_path)
        img_color = img.resize((resize_weidth, resize_height), Image.ANTIALIAS)#处理图像尺寸
        img_color.save(color_images_path + img_name, quality=95)#放入clolrimage
        print("Info: image '" + color_images_path + img_name + "' saved.")
        img_gray = img_color.convert('L')
        img_gray = img_gray.convert('RGB')
        img_gray.save(grayscale_images_path + img_name, quality=95)#转为灰色图像
        print("Info: image '" + grayscale_images_path + img_name + "' saved.")
        """
        下面这段代码可舍弃 如果不需要进行拼接
        """
        combined_image = Image.new('RGB', (resize_weidth*2, resize_height))#因为准备拼接故double size
        combined_image.paste(img_color, (0, 0, resize_weidth, resize_height))
        combined_image.paste(img_gray, (resize_weidth, 0, resize_weidth*2, resize_height))
        combined_image.save(combined_images_path + img_name, quality=95)
        print("Info: image '" + combined_images_path + img_name + "' saved.")
        pass