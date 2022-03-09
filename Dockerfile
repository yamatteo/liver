# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD [ "python", "-m" , "path_explorer"]
