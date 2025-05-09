FROM python:3.11.11-slim AS builder
WORKDIR /app

COPY ../src/client/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---

FROM python:3.11.11-slim
WORKDIR /app

ARG MODEL_NAME
ARG MODEL_TYPE

RUN test -n "$MODEL_NAME" || (echo "WARNING !!!!! MODEL_NAME is required" && exit 1)
RUN test -n "$MODEL_TYPE" || (echo "WARNING !!!!! MODEL_TYPE is required" && exit 1)

# Set as environment variables
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_TYPE=${MODEL_TYPE}

# Copy dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (including .env)
COPY ../src/client/app ./app
COPY ../src/client/joblib ./joblib

EXPOSE 8000
EXPOSE 8001

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

