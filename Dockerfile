# 1. Base Image (Lightweight Python)
FROM python:3.11-slim

# 2. Working Directory set karo
WORKDIR /app

# 3. Dependencies copy aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Baaki saara code aur static files copy karo
COPY . .

# 5. Port expose karo (FastAPI default 8000)
EXPOSE 8000

# 6. Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]