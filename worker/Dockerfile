FROM ubuntu:latest

RUN apt update && apt install -y python3 python3-pip

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

COPY . /worker
WORKDIR /worker
RUN pip3 install -r requirements.txt

ENTRYPOINT celery -A tasks worker --loglevel=info