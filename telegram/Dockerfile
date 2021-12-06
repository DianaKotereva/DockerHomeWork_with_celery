FROM ubuntu:latest

RUN apt update && apt install -y python3 python3-pip

ENV HOST '0.0.0.0'
ENV PORT 5002

COPY . /telegram
WORKDIR /telegram 
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 telegram.py