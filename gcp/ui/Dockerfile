FROM python:3.9-slim

WORKDIR /app

COPY ../src/client/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ui/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=/streamlit-app", "--server.enableXsrfProtection=false"]
