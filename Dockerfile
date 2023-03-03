# Dockerfile, Images, Container

# ARG PORT=6969
FROM python:3.9

ADD . /
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

CMD python -m uvicorn main:app --host 0.0.0.0 --port 80