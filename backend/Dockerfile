FROM python:3.11-slim

# set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=8080

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (for better caching)
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# expose port
EXPOSE 8080

# run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]