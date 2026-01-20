from faster_whisper import WhisperModel
from typing import Any, List, Tuple, Iterable, BinaryIO, Optional
from numpy.typing import NDArray
from numpy import dtype, ndarray
from pathlib import Path
from faster_whisper.transcribe import Segment
from faster_whisper.vad import VadOptions
from enum import Enum
import asyncio

class EnumModels(Enum):
    """Модели Whisper"""
    tiny = "tiny"
    tiny_en = "tiny.en"
    base = "base"
    base_en = "base.en"
    small = "small"
    small_en = "small.en"
    distil_small_en = "distil-small.en"
    medium = "medium"
    medium_en = "medium.en"
    distil_medium_en = "distil-medium.en"
    large_v1 = "large-v1"
    large_v2 = "large-v2"
    large_v3 = "large-v3"
    large = "large"
    distil_large_v2 = "distil-large-v2"
    distil_large_v3 = "distil-large-v3"
    large_v3_turbo = "large-v3-turbo"
    turbo = "turbo"

class EnumDevices(Enum):
    """Устройства для запуска модели"""
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    rocm = "rocm"
    vulkan = "vulkan"
    metal = "metal"
    opencl = "opencl"
    intel_fpga = "intel_fpga"

class EnumComputeTypes(Enum):
    """Типы вычислений"""
    default = "default"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"

class EnumTask(Enum):
    """Задачи модели"""
    transcribe = "transcribe"
    translate = "translate"

class EnumLanguages(Enum):
    """Языки модели"""
    af = "af"
    am = "am"
    ar = "ar"
    as_ = "as"
    az = "az"
    ba = "ba"
    be = "be"
    bg = "bg"
    bn = "bn"
    bo = "bo"
    br = "br"
    bs = "bs"
    ca = "ca"
    cs = "cs"
    cy = "cy"
    da = "da"
    de = "de"
    el = "el"
    en = "en"
    es = "es"
    et = "et"
    eu = "eu"
    fa = "fa"
    fi = "fi"
    fo = "fo"
    fr = "fr"
    gl = "gl"
    gu = "gu"
    ha = "ha"
    haw = "haw"
    he = "he"
    hi = "hi"
    hr = "hr"
    ht = "ht"
    hu = "hu"
    hy = "hy"
    id_ = "id"
    is_ = "is"
    it = "it"
    ja = "ja"
    jw = "jw"
    ka = "ka"
    kk = "kk"
    km = "km"
    kn = "kn"
    ko = "ko"
    la = "la"
    lb = "lb"
    ln = "ln"
    lo = "lo"
    lt = "lt"
    lv = "lv"
    mg = "mg"
    mi = "mi"
    mk = "mk"
    ml = "ml"
    mn = "mn"
    mr = "mr"
    ms = "ms"
    mt = "mt"
    my = "my"
    ne = "ne"
    nl = "nl"
    nn = "nn"
    no = "no"
    oc = "oc"
    pa = "pa"
    pl = "pl"
    ps = "ps"
    pt = "pt"
    ro = "ro"
    ru = "ru"
    sa = "sa"
    sd = "sd"
    si = "si"
    sk = "sk"
    sl = "sl"
    sn = "sn"
    so = "so"
    sq = "sq"
    sr = "sr"
    su = "su"
    sv = "sv"
    sw = "sw"
    ta = "ta"
    te = "te"
    tg = "tg"
    th = "th"
    tk = "tk"
    tl = "tl"
    tr = "tr"
    tt = "tt"
    uk = "uk"
    ur = "ur"
    uz = "uz"
    vi = "vi"
    yi = "yi"
    yo = "yo"
    zh = "zh"
    yue = "yue"

class FasterWhisperModel:
    """Модель Whisper"""
    instance = None

    def __new__(cls, *args, **kwargs):
        """Создание экземпляра модели (singleton)"""
        if cls.instance is None:
            cls.instance = super(FasterWhisperModel, cls).__new__(cls)
        return cls.instance
    
    def __init__(
            self, 
            model: Optional[EnumModels] = None, 
            model_size_or_path: Optional[EnumModels] = None, 
            device: EnumDevices = EnumDevices.cuda, 
            device_index: Optional[int | List[int]] = 0, 
            compute_type: EnumComputeTypes = EnumComputeTypes.default, 
            cpu_threads: int = 0, 
            num_workers: int = 1, 
            download_root: Optional[str | None] = None, 
            local_files_only: bool = False, 
            files: Optional[dict | None] = None, 
            revision: Optional[str | None] = None, 
            use_auth_token: Optional[str | bool | None] = None, 
            **model_kwargs: Optional[dict | Any]) -> None:

        """Инициализация модели"""
        
        # Инициализируем только если переданы параметры модели
        if model is not None and model_size_or_path is not None:
            self.model_name = model.value
            self.model_size_or_path = model_size_or_path.value
            self.device = device.value
            self.device_index = device_index
            self.compute_type = compute_type.value
            self.cpu_threads = cpu_threads
            self.num_workers = num_workers
            self.download_root = download_root
            self.local_files_only = local_files_only
            self.files = files
            self.revision = revision
            self.use_auth_token = use_auth_token
            self.model_kwargs = model_kwargs

            self.model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=self.download_root,
                local_files_only=self.local_files_only,
                files=self.files,
                revision=self.revision,
                use_auth_token=self.use_auth_token,
                **self.model_kwargs
            )
        elif not hasattr(self, 'model'):
            self.model = None
    
    async def update_settings(
            self, 
            model: Optional[EnumModels] = None, 
            model_size_or_path: Optional[EnumModels] = None, 
            device: Optional[EnumDevices] = None, 
            device_index: Optional[int | List[int]] = 0, 
            compute_type: Optional[EnumComputeTypes] = None, 
            cpu_threads: Optional[int] = None, 
            num_workers: Optional[int] = None, 
            download_root: Optional[str] = None, 
            local_files_only: Optional[bool] = None, 
            files: Optional[dict] = None, 
            revision: Optional[str] = None, 
            use_auth_token: Optional[str | bool] = None, 
            **model_kwargs: Optional[dict | Any]) -> None:

        """Обновление настроек модели"""
        
        if not hasattr(self, 'model_name') or model is not None:
            self.model_name = model.value if model is not None else getattr(self, 'model_name', None)
        if not hasattr(self, 'model_size_or_path') or model_size_or_path is not None:
            self.model_size_or_path = model_size_or_path.value if model_size_or_path is not None else getattr(self, 'model_size_or_path', None)
        if not hasattr(self, 'device') or device is not None:
            self.device = device.value if device is not None else getattr(self, 'device', EnumDevices.cpu.value)
        if not hasattr(self, 'device_index') or device_index is not None:
            self.device_index = device_index if device_index is not None else getattr(self, 'device_index', 0)
        if not hasattr(self, 'compute_type') or compute_type is not None:
            self.compute_type = compute_type.value if compute_type is not None else getattr(self, 'compute_type', EnumComputeTypes.default.value)
        if not hasattr(self, 'cpu_threads') or cpu_threads is not None:
            self.cpu_threads = cpu_threads if cpu_threads is not None else getattr(self, 'cpu_threads', 0)
        if not hasattr(self, 'num_workers') or num_workers is not None:
            self.num_workers = num_workers if num_workers is not None else getattr(self, 'num_workers', 1)
        if not hasattr(self, 'download_root') or download_root is not None:
            self.download_root = download_root if download_root is not None else getattr(self, 'download_root', None)
        if not hasattr(self, 'local_files_only') or local_files_only is not None:
            self.local_files_only = local_files_only if local_files_only is not None else getattr(self, 'local_files_only', False)
        if not hasattr(self, 'files') or files is not None:
            self.files = files if files is not None else getattr(self, 'files', None)
        if not hasattr(self, 'revision') or revision is not None:
            self.revision = revision if revision is not None else getattr(self, 'revision', None)
        if not hasattr(self, 'use_auth_token') or use_auth_token is not None:
            self.use_auth_token = use_auth_token if use_auth_token is not None else getattr(self, 'use_auth_token', None)
        if not hasattr(self, 'model_kwargs') or model_kwargs:
            self.model_kwargs = model_kwargs if model_kwargs else getattr(self, 'model_kwargs', {})

        if self.model_name is None or self.model_size_or_path is None:
            raise ValueError("Модель и model_size_or_path должны быть указаны")

        def _create_model():
            return WhisperModel(
                self.model_size_or_path,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=self.download_root,
                local_files_only=self.local_files_only,
                files=self.files,
                revision=self.revision,
                use_auth_token=self.use_auth_token,
                **self.model_kwargs
            )
        
        self.model = await asyncio.to_thread(_create_model)


    async def transcribe_settings(self, language: Optional[EnumLanguages] = None, task: str = "transcribe", log_progress: bool = False, beam_size: int = 5, best_of: int = 5, patience: float = 1, length_penalty: float = 1, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, temperature: Optional[float | List[float] | Tuple[float, ...]] = [0, 0.2, 0.4, 0.6, 0.8, 1], compression_ratio_threshold: Optional[float] = 2.4, log_prob_threshold: Optional[float] = -1, no_speech_threshold: Optional[float] = 0.6, condition_on_previous_text: bool = True, prompt_reset_on_temperature: float = 0.5, initial_prompt: Optional[str | Iterable[int]] = None, prefix: Optional[str] = None, suppress_blank: bool = True, suppress_tokens: Optional[List[int]] = [-1], without_timestamps: bool = False, max_initial_timestamp: float = 1, word_timestamps: bool = False, prepend_punctuations: str = "\"'“¿([{-", append_punctuations: str = "\"'.。,，!！?？:：”)]}、", multilingual: bool = False, vad_filter: bool = False, vad_parameters: Optional[dict | VadOptions] = None, max_new_tokens: Optional[int] = None, chunk_length: Optional[int] = None, clip_timestamps: Optional[str | List[float]] = "0", hallucination_silence_threshold: Optional[float] = None, hotwords: Optional[str] = None, language_detection_threshold: Optional[float] = 0.5, language_detection_segments: int = 1) -> None:
        """Настройки транскрибации"""
        language_value = language.value if language else None
        task_value = task.value if isinstance(task, EnumTask) else task

        self.call_settings = {
            "language": language_value,
            "task": task_value,
            "log_progress": log_progress,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "prompt_reset_on_temperature": prompt_reset_on_temperature,
            "initial_prompt": initial_prompt,
            "prefix": prefix,
            "suppress_blank": suppress_blank,
            "suppress_tokens": suppress_tokens,
            "without_timestamps": without_timestamps,
            "max_initial_timestamp": max_initial_timestamp,
            "word_timestamps": word_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations,
            "multilingual": multilingual,
            "vad_filter": vad_filter,
            "vad_parameters": vad_parameters,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "clip_timestamps": clip_timestamps,
            "hallucination_silence_threshold": hallucination_silence_threshold,
            "hotwords": hotwords,
            "language_detection_threshold": language_detection_threshold,
            "language_detection_segments": language_detection_segments,
        }

    async def transcribe(self, audio: str | BinaryIO | ndarray[Any, dtype[Any]]) -> tuple[List[Segment], dict[str, Any]]:
        """Транскрибация аудио"""
        
        if self.model is None:
            raise ValueError("Модель не инициализирована")
        
        if not hasattr(self, 'call_settings'):
            raise ValueError("Настройки транскрибации не установлены. Вызовите transcribe_settings() сначала.")

        def _transcribe():
            segments_generator, info = self.model.transcribe(audio=audio, **self.call_settings)
            segments_list = list(segments_generator)
            return (segments_list, info)

        return await asyncio.to_thread(_transcribe)