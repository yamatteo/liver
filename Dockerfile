# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt update
RUN apt install -y build-essential cmake git libgdcm-tools


COPY nifty_reg_build ./nifty_reg_build
COPY nifty_reg_source ./nifty_reg_source
COPY preprocessing ./preprocessing
COPY entry.sh options.py LICENSE path_explorer.py README.md requirements.txt ./
COPY saved_models /models
COPY docker_envs.py envs.py
COPY segm ./segm
COPY models ./models
COPY dataset ./dataset


RUN pip install -r requirements.txt

RUN export CFLAGS=" -g -O2 -lm -ldl -Wall -Wpointer-arith -finline-functions -ffast-math -funroll-all-loops"
RUN cd nifty_reg_build && make && make install

CMD [ "sh", "entry.sh" ]
