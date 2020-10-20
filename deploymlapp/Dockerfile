FROM python:3.7.3

WORKDIR /app 

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "index.py" ]