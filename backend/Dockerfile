FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf_cache
COPY backend/requirements.txt .
# CPU-only PyTorch keeps the image small; training happens on Colab, not here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
