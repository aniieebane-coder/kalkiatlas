FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY . .

CMD ["sh", "-c", "streamlit run playground/app.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true"]
