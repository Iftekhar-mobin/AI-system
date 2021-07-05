FROM ubuntu:18.04

RUN apt-get -y update \
    && apt-get upgrade -y \
    && apt-get install -y \
        python3 \
        python3-dev \
        python3-pip \
        curl \
        sudo \
    && pip3 install -U pip setuptools

WORKDIR /usr/src/app
COPY . .

CMD python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python3", "./app.py"]