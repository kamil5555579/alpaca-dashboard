FROM python:3.11
WORKDIR /alpaca-dashoard
COPY . .
RUN pip install -r requirements.txt
CMD gunicorn --bind 0.0.0.0:5000 app:server --workers 4 --timeout 120