FROM ubuntu:16.04

WORKDIR /opt/recsys

RUN apt-get update

RUN apt-get install -y build-essential make wget git vim python3.5 python3-pip python3-venv cron libssl-dev

RUN python3 -m pip install --upgrade pip setuptools wheel numpy

RUN git clone https://github.com/alexeygrigorev/libffm-python && cd libffm-python/ && make so && mv libffm.so ffm

ADD . /opt/recsys/

RUN pip3 install . && python3 -m pip install -r requirements.txt && python3 setup.py install && python3 -m nltk.downloader all

VOLUME [ "/opt/recsys/data" ]

EXPOSE 8080

#CMD [ "/opt/recsys/install.sh"]
CMD [ "/bin/bash"]
