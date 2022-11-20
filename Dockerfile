# 
# FROM python:3.9
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN apt update
RUN apt install -y git libsndfile1-dev python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# 
COPY ./app /code/app

EXPOSE 8789

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8789"]

