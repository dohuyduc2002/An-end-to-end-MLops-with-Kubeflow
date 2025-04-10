FROM python:3.10-slim

LABEL maintainer="ducdh2"
LABEL organization="microwave"

WORKDIR /mlflow/

ARG MLFLOW_VERSION

RUN apt-get update -y
RUN apt-get install -y iputils-ping
RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2-binary

EXPOSE 4000

# Ref: https://github.com/mlflow/mlflow/issues/6274#issuecomment-1196106372
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000

# docker build -t test_k1 -f virtualization/Dockerfile --build-arg MLFLOW_VERSION=2.3.2 . && docker run -p 5002:5000 test_k1
