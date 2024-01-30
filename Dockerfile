FROM tensorflow/tensorflow:latest-gpu

WORKDIR /project
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
