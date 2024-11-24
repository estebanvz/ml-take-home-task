FROM python:3.9-bullseye
RUN useradd -m -u 1000 user
RUN pip install --upgrade pip
RUN apt update -y
RUN apt install ffmpeg libsm6 libxext6  -y
RUN pip install tensorflow==2.15.0
RUN pip install numpy==1.26.4
RUN pip install matplotlib==3.9.2
RUN pip install transformers==4.46.3
RUN pip install torch==2.5.1
RUN pip install easyocr==1.7.2
RUN pip install pandas==2.2.3