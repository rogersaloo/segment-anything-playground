FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

COPY ./fine_tuning /model/

COPY ./app .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "app", "api.py"]