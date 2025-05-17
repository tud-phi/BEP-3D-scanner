FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache efficiency)
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

CMD [ "python", "watcher.py" ]