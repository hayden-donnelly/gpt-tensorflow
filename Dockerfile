FROM tensorflow/tensorflow:latest-gpu

WORKDIR /project
COPY requirements.txt requirements.txt

RUN apt-get -y update
RUN apt-get -y install git

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]