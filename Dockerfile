FROM ubuntu:18.04

MAINTAINER Wahyu Akbar

RUN apt-get -y update
    
# RUN apt-get -y upgrade

RUN apt-get -y install python3 python3-venv python3-dev

RUN apt-get -y install nginx git

RUN apt-get -y install git

RUN apt-get -y install libsm6 libxext6 libxrender-dev

ADD . /flask-app

WORKDIR /flask-app

RUN python3 -m venv venv

RUN source venv/bin/activate

RUN pip3 install -r requirements.txt

RUN export FLASK_APP=app.py

RUN flask run -h 0.0.0.0

# ENTRYPOINT ["python3"]

# CMD ["app.py"]
