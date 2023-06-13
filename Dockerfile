FROM ubuntu:latest
LABEL authors="archy"

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

WORKDIR /mydata

COPY data/ ./data
COPY src/ ./src
COPY requirements.txt .
COPY main.py .

RUN pip install -r requirements.txt
RUN cd src
RUN ls -al
RUN bash

ENTRYPOINT ["top", "-b"]