FROM python:3.11-slim

WORKDIR /app

# System deps for prophet/neuralprophet (pystan, cmdstanpy, compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only first (avoids pulling CUDA ~4GB variant)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile Stan models on build to avoid cold-start delay
RUN python -c "from prophet import Prophet; Prophet().fit(__import__('pandas').DataFrame({'ds':['2020-01-01'],'y':[1]}))" || true

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
