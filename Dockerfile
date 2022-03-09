# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt update
RUN apt install -y build-essential cmake git libgdcm-tools
COPY . .
RUN export CFLAGS=" -g -O2 -lm -ldl -Wall -Wpointer-arith -finline-functions -ffast-math -funroll-all-loops"
RUN cd nifty_reg_build && make && make install

CMD [ "sh", "entry.sh" ]
