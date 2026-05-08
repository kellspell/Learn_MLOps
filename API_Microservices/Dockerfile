FROM python:3.11-slim
WORKDIR /app
COPY ./app /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "weather_api:app", "--host", "0.0.0.0", "--port", "80"]
