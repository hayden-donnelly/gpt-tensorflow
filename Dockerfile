FROM tensorflow/tensorflow:latest-gpu

WORKDIR .
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "model.py"]