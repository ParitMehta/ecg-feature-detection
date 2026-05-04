FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app  

RUN pip install --no-cache-dir \
    "streamlit>=1.35" \
    "jinja2>=3.1,<4" \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "scipy>=1.11" \
    "matplotlib>=3.7" \
    "wfdb>=4.1" \
    "requests>=2.31"

COPY scripts/ECG_Classifier.py scripts/ECG_Classifier.py
COPY scripts/pages/ scripts/pages/
COPY src/paths.py src/paths.py
COPY src/__init__.py src/__init__.py
COPY data/processed/X_test.npy data/processed/X_test.npy
COPY data/processed/y_test.npy data/processed/y_test.npy
COPY data/processed/class_names.txt data/processed/class_names.txt
COPY reports/train/thresholds.json reports/train/thresholds.json
COPY reports/fairness/ reports/fairness/

EXPOSE 8501
CMD ["streamlit", "run", "scripts/ECG_Classifier.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
