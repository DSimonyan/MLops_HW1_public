FROM python:3.9

COPY requirements.txt .
RUN python -m pip install -r requirements.txt


EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV RUNTIME_DOCKER Yes

COPY . .