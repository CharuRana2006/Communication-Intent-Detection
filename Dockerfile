
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet omw-1.4

ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
