FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY 01_your_canada_version/requirements-local.txt 01_your_canada_version/requirements-local.txt

RUN pip install --upgrade pip \
    && pip install -r 01_your_canada_version/requirements-local.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run 01_your_canada_version/app/app_local.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --browser.gatherUsageStats=false"]
