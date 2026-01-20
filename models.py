from typing import Optional, List, Any, Iterable, Tuple
from pydantic import BaseModel, Field, ConfigDict
from whisper import EnumModels, EnumDevices, EnumComputeTypes, EnumTask, EnumLanguages
from faster_whisper.vad import VadOptions

class SettingsRequest(BaseModel):
    """Запрос на установку настроек модели"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "large-v3",
                    "device": "cuda",
                    "device_index": 0,
                    "compute_type": "float16",
                    "cpu_threads": 4,
                    "num_workers": 1
                },
                {
                    "model": "base",
                    "device": "cpu",
                    "device_index": 0,
                    "compute_type": "int8",
                    "cpu_threads": 8,
                    "num_workers": 2
                },
                {
                    "model": "large-v3-turbo",
                    "device": "mps",
                    "device_index": [0, 1],
                    "compute_type": "default",
                    "cpu_threads": 0,
                    "num_workers": 1,
                    "download_root": "/path/to/models"
                }
            ]
        }
    )

    model: Optional[EnumModels] = Field(default=EnumModels.large_v3, description="Модель Whisper", examples=[EnumModels.large_v3, EnumModels.base, EnumModels.small])
    device: EnumDevices = Field(default=EnumDevices.cuda, description="Устройство для запуска модели", examples=[EnumDevices.cuda, EnumDevices.cpu, EnumDevices.mps])
    device_index: Optional[int | List[int]] = Field(default=0, description="Индекс устройства для запуска модели", examples=[0, [0, 1]])
    compute_type: EnumComputeTypes = Field(default=EnumComputeTypes.default, description="Тип вычислений", examples=[EnumComputeTypes.default, EnumComputeTypes.float16, EnumComputeTypes.int8])
    cpu_threads: int = Field(default=0, description="Количество потоков для CPU", examples=[0, 4, 8])
    num_workers: int = Field(default=1, description="Количество потоков для запуска модели", examples=[1, 2, 4])
    download_root: Optional[str | None] = Field(default=None, description="Папка для загрузки модели", examples=["/path/to/models", None])
    local_files_only: bool = Field(default=False, description="Флаг для загрузки модели из локальной папки", examples=[False, True])
    files: Optional[dict | None] = Field(default=None, description="Файлы модели", examples=[None, {"model.bin": "path"}])
    revision: Optional[str | None] = Field(default=None, description="Версия модели", examples=[None, "main", "v1.0"])
    use_auth_token: Optional[str | bool | None] = Field(default=None, description="Токен для загрузки модели", examples=[None, "hf_token_here", True])
    model_kwargs: Optional[dict | Any] = Field(default=None, description="Параметры модели", examples=[None, {"key": "value"}])


class TranscriptionSettings(BaseModel):
    """Настройки для транскрибации аудио"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "language": "ru",
                    "task": "transcribe",
                    "temperature": 0.0,
                    "beam_size": 5,
                    "vad_filter": True
                },
                {
                    "language": "en",
                    "task": "translate",
                    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "word_timestamps": True,
                    "vad_filter": False
                }
            ]
        }
    )

    language: Optional[EnumLanguages] = Field(default=None, description="Язык модели", examples=[EnumLanguages.ru, EnumLanguages.en, EnumLanguages.de, None])
    task: EnumTask = Field(default=EnumTask.transcribe, description="Задача модели", examples=[EnumTask.transcribe, EnumTask.translate])
    log_progress: bool = Field(default=False, description="Флаг для вывода прогресса", examples=[False, True])
    beam_size: int = Field(default=5, description="Размер луча", examples=[1, 5, 10])
    best_of: int = Field(default=5, description="Количество лучей", examples=[1, 5, 10])
    patience: float = Field(default=1, description="Пациентность", examples=[1.0, 1.5, 2.0])
    length_penalty: float = Field(default=1, description="Штраф за длину", examples=[1.0, 1.2, 1.5])
    repetition_penalty: float = Field(default=1, description="Штраф за повторение", examples=[1.0, 1.2, 1.5])
    no_repeat_ngram_size: int = Field(default=0, description="Размер n-грамма для исключения повторений", examples=[0, 3, 5])
    temperature: Optional[float | List[float] | Tuple[float, ...]] = Field(default=None, description="Температура", examples=[0.0, 0.5, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    compression_ratio_threshold: Optional[float] = Field(default=2.4, description="Порог сжатия", examples=[2.4, 3.0, None])
    log_prob_threshold: Optional[float] = Field(default=-1, description="Порог логарифма вероятности", examples=[-1.0, -0.5, None])
    no_speech_threshold: Optional[float] = Field(default=0.6, description="Порог беззвучности", examples=[0.6, 0.8, None])
    condition_on_previous_text: bool = Field(default=True, description="Флаг для условия на предыдущий текст", examples=[False, True])
    prompt_reset_on_temperature: float = Field(default=0.5, description="Температура для сброса prompt", examples=[0.5, 1.0])
    initial_prompt: Optional[str | Iterable[int]] = Field(default=None, description="Prompt для начала транскрибации", examples=[None, "Привет, это тест"])
    prefix: Optional[str] = Field(default=None, description="Префикс для транскрибации", examples=[None, "Текст начинается с"])
    suppress_blank: bool = Field(default=True, description="Флаг для исключения пустых сегментов", examples=[True, False])
    suppress_tokens: Optional[List[int]] = Field(default=[-1], description="Токены для исключения", examples=[[-1], [1, 2, 3], None])
    without_timestamps: bool = Field(default=False, description="Флаг для исключения временных меток", examples=[False, True])
    max_initial_timestamp: float = Field(default=1, description="Максимальное время начала транскрибации", examples=[1.0, 2.0])
    word_timestamps: bool = Field(default=False, description="Флаг для вывода временных меток слов", examples=[False, True])
    prepend_punctuations: str = Field(default="\"'“¿([{-", description="Префиксы для транскрибации", )
    append_punctuations: str = Field(default="\"'.。,，!！?？:：”)]}、", description="Суффиксы для транскрибации", )
    multilingual: bool = Field(default=False, description="Флаг для многоязычности", examples=[False, True])
    vad_filter: bool = Field(default=False, description="Флаг для фильтрации беззвучных сегментов", examples=[False, True])
    vad_parameters: Optional[dict | VadOptions] = Field(default=None, description="Параметры фильтрации беззвучных сегментов", examples=[None, {"threshold": 0.5}])
    max_new_tokens: Optional[int] = Field(default=None, description="Максимальное количество токенов", examples=[None, 100, 200])
    chunk_length: Optional[int] = Field(default=None, description="Длина чанка", examples=[None, 30, 60])
    clip_timestamps: Optional[str | List[float]] = Field(default="0", description="Временные метки для обрезки", examples=["0", [0.0, 10.0], "0:10"])
    hallucination_silence_threshold: Optional[float] = Field(default=None, description="Порог беззвучности для hallucination", examples=[None, 0.5, 1.0])
    hotwords: Optional[str] = Field(default=None, description="Горячие слова для транскрибации", examples=[None, "слово1 слово2"])
    language_detection_threshold: Optional[float] = Field(default=0.5, description="Порог для определения языка", examples=[0.5, 0.7, 0.9])
    language_detection_segments: int = Field(default=1, description="Количество сегментов для определения языка", examples=[1, 3, 5])

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

class StatusResponse(BaseModel):
    """Ответ со статусом операции"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": True,
                    "message": "Настройки успешно установлены"
                },
                {
                    "status": False,
                    "message": "Ошибка при загрузке модели"
                }
            ]
        }
    )

    status: bool = Field(default=True, description="Статус ответа", examples=[True, False])
    message: Optional[str] = Field(default=None, description="Сообщение", examples=["Успешно", "Ошибка", None])


class TranscriptionResponse(BaseModel):
    """Ответ с результатом транскрибации"""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": True,
                    "message": "Транскрибация завершена",
                    "data": {
                        "text": "Привет, это тест транскрибации",
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 2.5,
                                "text": "Привет, это тест транскрибации"
                            }
                        ]
                    }
                },
                {
                    "status": False,
                    "message": "Ошибка при обработке аудио",
                    "data": None
                }
            ]
        }
    )

    status: bool = Field(default=True, description="Статус ответа", examples=[True, False])
    message: Optional[str] = Field(default=None, description="Сообщение", examples=["Транскрибация завершена", "Ошибка", None])
    data: Optional[dict[str, Any]] = Field(default=None, description="Данные", examples=[{"text": "текст"}, {"segments": []}, None])