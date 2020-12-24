FROM baturayo/tensorflow2.1.0-py3.7.7:latest

RUN mkdir /code
WORKDIR /code

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

RUN pip install pip -U -i https://mirrors.aliyun.com/pypi/simple/
ADD requirements.txt /code/
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

ADD . /code/

EXPOSE 1234
CMD ["python", "manage.py", "migrate"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:1234"]
