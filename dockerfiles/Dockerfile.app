FROM python:3.11.11-slim AS builder
WORKDIR /app

COPY ./requirements_api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11.11-slim
WORKDIR /app

ARG MODEL_NAME
ARG MODEL_TYPE

ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_TYPE=${MODEL_TYPE}
ENV PYTHONPATH="/app"      

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/client ./client

EXPOSE 8000
EXPOSE 8001

CMD ["uvicorn", "client.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
