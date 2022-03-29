# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt update
RUN apt install -y build-essential cmake git libgdcm-tools

COPY nifty_reg_build .
COPY nifty_reg_source .
COPY preprocessing .
RUN export CFLAGS=" -g -O2 -lm -ldl -Wall -Wpointer-arith -finline-functions -ffast-math -funroll-all-loops"
RUN cd nifty_reg_build && make && make install

COPY entry.sh LICENSE path_explorer.py README.md requirements.txt ./
RUN pip install -r requirements.txt

COPY saved_models /models
CMD [ "sh", "entry.sh" ]
