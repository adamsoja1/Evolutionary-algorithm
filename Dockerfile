FROM ubuntu:22.04

RUN apt-get -y update && apt-get -y install python3

RUN apt-get install -y python3-pip
RUN pip install --upgrade pip




RUN mkdir -p /python

WORKDIR /python

COPY . /python

RUN pip3 install -r requirements.txt

CMD python3 Evolutionary_algorithm.py
