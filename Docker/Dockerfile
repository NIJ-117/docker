# 1️⃣ Use a lightweight Python base image
FROM python:3.9-slim

# 2️⃣ Install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    espeak-ng \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3️⃣ Upgrade pip and install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4️⃣ Copy the application files
WORKDIR /app
COPY . /app

# 5️⃣ Install Git LFS and clone Hugging Face repo
RUN git lfs install && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M && \
    mv Kokoro-82M/* . && \
    rm -rf Kokoro-82M

# 6️⃣ Expose port 8000 and run FastAPI app
EXPOSE 8000
CMD ["uvicorn", "generate:app", "--host", "0.0.0.0", "--port", "8000"]
