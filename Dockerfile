FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg git build-essential curl \
    libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip до последней версии через get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - && \
    python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
# Устанавливаем зависимости (после обновления pip поддерживает --break-system-packages)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

EXPOSE 8000

# Запуск приложения
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]