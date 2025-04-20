# Stage 1: Build dependencies
FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim
WORKDIR /app

# 🔧 Add debug tools
RUN apt-get update && apt-get install -y \
    curl \
    iputils-ping \
    lsof \
    net-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app/ ./app
EXPOSE 8000
EXPOSE 8001
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
