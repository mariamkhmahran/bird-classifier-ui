FROM ubuntu
FROM python:3.6
FROM tensorflow/tensorflow:2.4.3

ADD . /

RUN apt-get update
RUN pip install -r requirements.txt

WORKDIR /

CMD [ "python", "app.py"]