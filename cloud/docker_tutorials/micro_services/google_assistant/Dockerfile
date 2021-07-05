FROM ubuntu:18.04

RUN apt-get -y update \
    && apt-get upgrade -y \
    && apt-get install -y \
        locales \
        python3 \
        python3-dev \
        python3-pip \
        curl \
        sudo \
    && pip3 install -U pip setuptools

RUN sudo locale-gen ja_JP.UTF-8
RUN sudo chmod 0755 /etc/default/locale

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP.UTF-8

WORKDIR /usr/src/app
COPY . .

CMD python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8004

CMD ["python3", "./app.py"]