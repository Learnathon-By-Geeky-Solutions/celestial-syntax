FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    cmake gcc g++ make pkg-config \
    libgl1 libglu1-mesa libsm6 libice6 libxrender1 libxext6 libx11-6 libxcb1 libxi6 libxt6 libxfixes3 libxau6 libxdmcp6 libuuid1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
