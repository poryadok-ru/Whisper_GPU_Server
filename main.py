from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from models import SettingsRequest, TranscriptionSettings, StatusResponse, TranscriptionResponse
from whisper import FasterWhisperModel, EnumModels, EnumDevices, EnumComputeTypes, EnumTask, EnumLanguages

app = FastAPI()

@app.get(
    "/health",
    response_model=StatusResponse,
    status_code=200,
    response_description="Статус сервиса",
    description="Проверка работоспособности сервиса",
    summary="Health check",
    tags=["health"],
)
async def health_check():
    """Проверка работоспособности сервиса"""
    try:
        model_instance = FasterWhisperModel()
        model_status = model_instance.model is not None if hasattr(model_instance, 'model') else False
        
        return StatusResponse(
            status=True,
            message=f"Сервис работает. Модель инициализирована: {model_status}"
        )
    except Exception as e:
        return StatusResponse(
            status=False,
            message=f"Ошибка при проверке здоровья сервиса: {str(e)}"
        )

@app.post(
    "/set_settings_model", 
    response_model=StatusResponse, 
    status_code=200, 
    response_description="Успешно установлены настройки модели",
    description="Установка настроек модели", 
    summary="Установка настроек модели",
    tags=["whisper"],
)
async def set_settings_model(settings: SettingsRequest):
    try:
        model = FasterWhisperModel(
            settings.model, 
            settings.model,  
            settings.device, 
            settings.device_index, 
            settings.compute_type, 
            settings.cpu_threads, 
            settings.num_workers, 
            settings.download_root, 
            settings.local_files_only, 
            settings.files, 
            settings.revision, 
            settings.use_auth_token, 
            **(settings.model_kwargs or {})
        )
        return StatusResponse(status=True, message="Настройки успешно установлены")
    
    except Exception as e:
        return StatusResponse(status=False, message=str(e))


@app.post(
    "/set_settings_transcription",
    response_model=StatusResponse,
    status_code=200,
    response_description="Успешно установлены настройки транскрибации",
    description="Установка настроек транскрибации",
    summary="Установка настроек транскрибации",
    tags=["whisper"],
)
async def set_settings_transcription(settings: TranscriptionSettings):
    try:
        model_instance = FasterWhisperModel()
        await model_instance.transcribe_settings(
            language=settings.language,
            task=settings.task.value if settings.task else "transcribe",
            log_progress=settings.log_progress,
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            patience=settings.patience,
            length_penalty=settings.length_penalty,
            repetition_penalty=settings.repetition_penalty,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            temperature=settings.temperature,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            log_prob_threshold=settings.log_prob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
            condition_on_previous_text=settings.condition_on_previous_text,
            prompt_reset_on_temperature=settings.prompt_reset_on_temperature,
            initial_prompt=settings.initial_prompt,
            prefix=settings.prefix,
            suppress_blank=settings.suppress_blank,
            suppress_tokens=settings.suppress_tokens,
            without_timestamps=settings.without_timestamps,
            max_initial_timestamp=settings.max_initial_timestamp,
            word_timestamps=settings.word_timestamps,
            prepend_punctuations=settings.prepend_punctuations,
            append_punctuations=settings.append_punctuations,
            multilingual=settings.multilingual,
            vad_filter=settings.vad_filter,
            vad_parameters=settings.vad_parameters,
            max_new_tokens=settings.max_new_tokens,
            chunk_length=settings.chunk_length,
            clip_timestamps=settings.clip_timestamps,
            hallucination_silence_threshold=settings.hallucination_silence_threshold,
            hotwords=settings.hotwords,
            language_detection_threshold=settings.language_detection_threshold,
            language_detection_segments=settings.language_detection_segments
        )
        return StatusResponse(status=True, message="Настройки успешно установлены")

    except Exception as e:
        return StatusResponse(status=False, message=str(e))


@app.post(
    "/set_base_settings_model",
    response_model=StatusResponse,
    status_code=200,
    response_description="Успешно установлены базовые настройки",
    description="Установка базовых настроек",
    summary="Установка базовых настроек",
    tags=["whisper"],
)
async def set_base_settings_model():
    try:
        model_instance = FasterWhisperModel()
        await model_instance.update_settings(
            model=EnumModels.base, 
            model_size_or_path=EnumModels.base, 
            device=EnumDevices.cpu, 
            device_index=0, 
            compute_type=EnumComputeTypes.default, 
            cpu_threads=4, 
            num_workers=1
        )
        return StatusResponse(status=True, message="Базовые настройки успешно установлены")
    except Exception as e:
        return StatusResponse(status=False, message=str(e))

@app.post(
    "/set_base_settings_transcription",
    response_model=StatusResponse,
    status_code=200,
    response_description="Успешно установлены базовые настройки",
    description="Установка базовых настроек",
    summary="Установка базовых настроек",
    tags=["whisper"],
)
async def set_base_settings_transcription():
    try:
        model_instance = FasterWhisperModel()
        await model_instance.transcribe_settings(language=EnumLanguages.ru, task="transcribe", log_progress=False, beam_size=5, best_of=5, patience=1, length_penalty=1, repetition_penalty=1, no_repeat_ngram_size=0, temperature=None, compression_ratio_threshold=2.4, log_prob_threshold=-1, no_speech_threshold=0.6, condition_on_previous_text=True, prompt_reset_on_temperature=0.5, initial_prompt=None, prefix=None, suppress_blank=True, suppress_tokens=[-1], without_timestamps=False, max_initial_timestamp=1, word_timestamps=False, prepend_punctuations="\"'“¿([{-", append_punctuations="\"'.。,，!！?？:：”)]}、", multilingual=False, vad_filter=False, vad_parameters=None, max_new_tokens=None, chunk_length=None, clip_timestamps="0", hallucination_silence_threshold=None, hotwords=None, language_detection_threshold=0.5, language_detection_segments=1)
        return StatusResponse(status=True, message="Базовые настройки успешно установлены")
    except Exception as e:
        return StatusResponse(status=False, message=str(e))

@app.post(
    "/transcribe_audio",
    response_model=TranscriptionResponse,
    status_code=200,
    response_description="Успешно выполнена транскрибация",
    description="Транскрибация аудио",
    summary="Транскрибация аудио",
    tags=["whisper"],
)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        model_instance = FasterWhisperModel()
        segments, info = await model_instance.transcribe(file.file)
        
        segments_data = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            for segment in segments
        ]
        
        full_text = " ".join([segment.text for segment in segments])
        
        language = getattr(info, 'language', None)
        language_probability = getattr(info, 'language_probability', None)
        
        return TranscriptionResponse(
            status=True, 
            message="Транскрибация успешно выполнена", 
            data={
                "text": full_text,
                "segments": segments_data,
                "language": language,
                "language_probability": language_probability
            }
        )
    
    except Exception as e:
        return TranscriptionResponse(status=False, message=str(e))