FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the code and the dataset
COPY . .

# Train inside the image: the saved model then always matches the library versions installed here
RUN python train_model.py

EXPOSE 8000

# SERVICE=api (default) starts the FastAPI backend, SERVICE=ui starts the Streamlit frontend
CMD ["python", "run.py"]
