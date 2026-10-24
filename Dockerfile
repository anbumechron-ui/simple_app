
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ingest.py spotify.pdf ./

EXPOSE 8501

CMD ["sh", "-c", "if [ ! -f data/faiss_index/index.faiss ]; then python ingest.py; fi && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
