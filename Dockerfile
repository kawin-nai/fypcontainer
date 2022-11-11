FROM python:3.8

WORKDIR /azure-flask

COPY requirements.txt .

RUN apt-get update

RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/main.py"]