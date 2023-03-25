FROM tensorflow/tensorflow:latest-gpu

WORKDIR .
COPY requirements.txt requirements.txt

RUN apt-get -y update
RUN apt-get -y install git

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . ./gpt-tensorflow

cmd ["python", "-m", "spacy", "download", "en_core_web_sm"]