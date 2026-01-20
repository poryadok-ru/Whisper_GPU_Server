# Whisper GPU Server

GPU-сервер распознавания речи (ASR) на базе OpenAI Whisper для обработки аудио и транскрибации звонков.

## 🚀 Возможности

- **Высокопроизводительная транскрибация** с использованием faster-whisper
- **Поддержка GPU** (CUDA) для ускорения обработки
- **REST API** на базе FastAPI
- **Множество моделей** Whisper (от tiny до large-v3)
- **Гибкие настройки** транскрибации (язык, температура, beam size и др.)
- **Docker поддержка** для легкого развертывания

## 📋 Требования

- Python 3.11+
- CUDA 12.2+ (для GPU) или CPU
- Docker (опционально, для контейнеризации)
- NVIDIA GPU с поддержкой CUDA (рекомендуется для production)

## 🔧 Установка

### Локальная установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd Whisper_GPU_Server
```

2. Создайте виртуальное окружение:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # На Windows: .venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Запустите сервер:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker установка

1. Соберите образ:
```bash
docker build -t whisper-gpu-server .
```

2. Запустите контейнер с GPU:
```bash
docker run --gpus all -p 8000:8000 whisper-gpu-server
```

Или с docker-compose (создайте `docker-compose.yml`):
```yaml
version: '3.8'
services:
  whisper:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 📚 API Документация

После запуска сервера документация доступна по адресам:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔌 API Endpoints

### 1. Health Check

Проверка работоспособности сервиса.

**GET** `/health`

**Ответ:**
```json
{
  "status": true,
  "message": "Сервис работает. Модель инициализирована: True"
}
```

### 2. Установка настроек модели

Инициализация или обновление настроек модели Whisper.

**POST** `/set_settings_model`

**Тело запроса:**
```json
{
  "model": "large-v3",
  "device": "cuda",
  "device_index": 0,
  "compute_type": "float16",
  "cpu_threads": 4,
  "num_workers": 1
}
```

**Доступные модели:**
- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v1`, `large-v2`, `large-v3`
- `large-v3-turbo`, `turbo`

**Устройства:**
- `cpu` - CPU обработка
- `cuda` - NVIDIA GPU
- `mps` - Apple Silicon GPU

**Типы вычислений:**
- `default` - автоматический выбор
- `float16` - для GPU
- `float32` - для CPU
- `int8` - квантованная модель

### 3. Установка базовых настроек модели

Быстрая инициализация с предустановленными параметрами.

**POST** `/set_base_settings_model`

Использует модель `base` на CPU (или можно изменить в коде на GPU).

### 4. Установка настроек транскрибации

Настройка параметров транскрибации (язык, температура и др.).

**POST** `/set_settings_transcription`

**Тело запроса:**
```json
{
  "language": "ru",
  "task": "transcribe",
  "temperature": 0.0,
  "beam_size": 5,
  "word_timestamps": false,
  "vad_filter": true
}
```

**Параметры:**
- `language` - код языка (ru, en, de и др.) или `null` для автоопределения
- `task` - `transcribe` (транскрибация) или `translate` (перевод на английский)
- `temperature` - температура генерации (0.0-1.0 или массив)
- `beam_size` - размер луча поиска (1-10)
- `word_timestamps` - включить временные метки для слов
- `vad_filter` - фильтрация беззвучных сегментов

### 5. Установка базовых настроек транскрибации

Быстрая настройка с русским языком.

**POST** `/set_base_settings_transcription`

### 6. Транскрибация аудио

Основной endpoint для транскрибации аудиофайлов.

**POST** `/transcribe_audio`

**Формат запроса:** `multipart/form-data`

**Параметры:**
- `file` - аудиофайл (WAV, MP3, M4A, FLAC и др.)

**Ответ:**
```json
{
  "status": true,
  "message": "Транскрибация успешно выполнена",
  "data": {
    "text": "Полный текст транскрибации",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Фрагмент текста"
      }
    ],
    "language": "ru",
    "language_probability": 0.99
  }
}
```

## 💡 Примеры использования

### cURL

```bash
# 1. Проверка здоровья
curl http://localhost:8000/health

# 2. Инициализация модели
curl -X POST http://localhost:8000/set_base_settings_model \
  -H "Content-Type: application/json"

# 3. Установка настроек транскрибации
curl -X POST http://localhost:8000/set_base_settings_transcription \
  -H "Content-Type: application/json"

# 4. Транскрибация аудио
curl -X POST http://localhost:8000/transcribe_audio \
  -F "file=@audio.wav"
```

### Python

```python
import requests

# Инициализация модели
response = requests.post(
    "http://localhost:8000/set_base_settings_model"
)
print(response.json())

# Настройка транскрибации
response = requests.post(
    "http://localhost:8000/set_base_settings_transcription"
)
print(response.json())

# Транскрибация
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe_audio",
        files={"file": f}
    )
    result = response.json()
    print(result["data"]["text"])
```

### JavaScript/TypeScript

```typescript
// Инициализация
await fetch('http://localhost:8000/set_base_settings_model', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' }
});

// Транскрибация
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch('http://localhost:8000/transcribe_audio', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.data.text);
```

## 🏗️ Структура проекта

```
Whisper_GPU_Server/
├── main.py              # FastAPI приложение и endpoints
├── whisper.py           # Класс FasterWhisperModel и настройки
├── models.py            # Pydantic модели для запросов/ответов
├── requirements.txt     # Python зависимости
├── Dockerfile           # Docker конфигурация
├── README.md           # Документация
└── test_transcribe.py  # Тестовый скрипт
```

## ⚙️ Конфигурация

### Изменение базовых настроек

В файле `main.py` можно изменить базовые настройки:

```python
# В функции set_base_settings_model()
await model_instance.update_settings(
    model=EnumModels.large_v3,  # Изменить модель
    device=EnumDevices.cuda,     # Изменить устройство
    compute_type=EnumComputeTypes.float16,
    cpu_threads=4,
    num_workers=1
)
```

## 🐛 Устранение неполадок

### Модель не инициализируется

1. Убедитесь, что модель загружена:
```bash
curl -X POST http://localhost:8000/set_base_settings_model
```

2. Проверьте статус:
```bash
curl http://localhost:8000/health
```

### Ошибки GPU

- Убедитесь, что CUDA установлена: `nvidia-smi`
- Для Docker используйте `--gpus all`
- Если GPU недоступна, используйте `device: "cpu"`

### Проблемы с памятью

- Используйте меньшую модель (например, `base` вместо `large-v3`)
- Уменьшите `num_workers`
- Используйте `int8` compute_type для экономии памяти