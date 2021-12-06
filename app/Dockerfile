FROM ubuntu:latest

RUN apt update && apt install -y python3 python3-pip

ENV HOST '0.0.0.0'
ENV PORT 5000
ENV DEBUG false
ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 cli.py