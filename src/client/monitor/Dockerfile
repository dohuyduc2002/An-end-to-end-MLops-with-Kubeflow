# Single‐stage is fine here
FROM python:3.9-slim
WORKDIR /app

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the monitor script
COPY monitor/monitor.py ./

EXPOSE 8001
CMD ["python", "monitor.py"]
