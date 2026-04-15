"""
transcriber.py
Dual-model ASR engine.
  - StreamingTranscriber: Parakeet v3 near-real-time chunks (dictation default)
  - OfflineTranscriber:   Parakeet v3 batch (meeting post-processing)
"""
import os
import json
import queue
import re
import threading
import urllib.error
import urllib.request
import wave
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sherpa_onnx


# ---------------------------------------------------------------------------
# Paths – override via env vars if needed
# ---------------------------------------------------------------------------
MODEL_DIR_STREAMING = os.environ.get(
    "FINCH_DICTATION_MODEL",
    "./models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
)
MODEL_DIR_OFFLINE = os.environ.get(
    "FINCH_OFFLINE_MODEL",
    "./models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
)
DICTATION_RULES_PATH = os.environ.get("FINCH_DICTATION_RULES", "./dictation_rules.json")
VAD_MODEL_PATH = os.environ.get("FINCH_SILERO_VAD_MODEL", "./models/silero_vad.int8.onnx")
PUNCTUATION_MODEL_DIR = os.environ.get(
    "FINCH_PUNCTUATION_MODEL_DIR",
    "./models/sherpa-onnx-online-punct-en-2024-08-06",
)
LANGUAGE_ID_MODEL_DIR = os.environ.get(
    "FINCH_LANGUAGE_ID_MODEL_DIR",
    "./models/sherpa-onnx-whisper-tiny",
)


DEFAULT_FILLER_WORDS = [
    "mm",
    "mmm",
    "um",
    "uh",
    "erm",
    "er",
    "ah",
]
DEFAULT_CUSTOM_WORDS = {
    "api": "API",
    "apis": "APIs",
    "cpu": "CPU",
    "gpu": "GPU",
    "ip": "IP",
    "ips": "IPs",
    "llm": "LLM",
    "ui": "UI",
    "url": "URL",
    "urls": "URLs",
    "vad": "VAD",
}
DEFAULT_CUSTOM_PHRASES = {}
DEFAULT_QUESTION_STARTERS = [
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "how",
    "is",
    "should",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "would",
]
DEFAULT_ALLOWED_LANGUAGES = ["en", "english"]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_language(language: str | None) -> str:
    if not language:
        return ""
    language = language.strip().lower()
    language = language.replace("<|", "").replace("|>", "")
    return language.split("-", 1)[0]


class PunctuationRestorer:
    def __init__(self, model_dir: str = PUNCTUATION_MODEL_DIR):
        self.punct = None
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.int8.onnx"
        self.vocab_path = self.model_dir / "bpe.vocab"
        self._initialize()

    def restore(self, text: str) -> str:
        if not self.punct or not text:
            return text

        bare = re.sub(r"\s+", " ", text).strip()
        bare = re.sub(r"[,.!?;:]+", "", bare)
        if not bare:
            return text

        try:
            return self.punct.add_punctuation_with_case(bare)
        except Exception as e:
            print(f"[Dictation] Punctuation restore failed: {e}")
            self.punct = None
            return text

    def _initialize(self):
        if not (self.model_path.exists() and self.vocab_path.exists()):
            return

        try:
            config = sherpa_onnx.OnlinePunctuationConfig(
                model_config=sherpa_onnx.OnlinePunctuationModelConfig(
                    cnn_bilstm=str(self.model_path),
                    bpe_vocab=str(self.vocab_path),
                    num_threads=1,
                    provider="cpu",
                )
            )
            self.punct = sherpa_onnx.OnlinePunctuation(config)
        except Exception as e:
            print(f"[Dictation] Punctuation model disabled: {e}")
            self.punct = None


class LLMTextRefiner:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.enabled = _truthy(os.environ.get("FINCH_LLM_POSTPROCESS"), False)
        self.enabled = bool(config.get("enabled", self.enabled))
        self.endpoint = str(config.get("endpoint") or os.environ.get("FINCH_LLM_ENDPOINT", "")).strip()
        self.model = str(config.get("model") or os.environ.get("FINCH_LLM_MODEL", "")).strip()
        self.api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        self.timeout_s = float(config.get("timeout_s") or os.environ.get("FINCH_LLM_TIMEOUT_S", "2.5"))

    def refine(self, text: str) -> str:
        if not (self.enabled and self.endpoint and self.model and text):
            return text

        prompt = (
            "Fix grammar, punctuation, capitalization, and obvious ASR mistakes. "
            "Preserve meaning. Return only corrected text."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(self.endpoint, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"[Dictation] LLM post-process skipped: {e}")
            return text

        refined = self._extract_text(result)
        return refined if refined else text

    @staticmethod
    def _extract_text(result: dict) -> str:
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return result["message"]["content"].strip()
        except (KeyError, TypeError):
            return ""


class LanguageValidator:
    def __init__(self, model_dir: str = LANGUAGE_ID_MODEL_DIR):
        self.identifier = None
        self.model_dir = Path(model_dir)
        self.encoder = self.model_dir / "tiny-encoder.int8.onnx"
        self.decoder = self.model_dir / "tiny-decoder.int8.onnx"
        self._initialize()

    def allowed(self, samples, sample_rate: int, allowed_languages) -> bool:
        language = self.detect(samples, sample_rate)
        if not language:
            return True
        allowed = {_clean_language(language) for language in allowed_languages}
        allowed.discard("")
        return not allowed or _clean_language(language) in allowed

    def detect(self, samples, sample_rate: int) -> str:
        if not self.identifier or len(samples) < sample_rate:
            return ""

        try:
            stream = self.identifier.create_stream()
            stream.accept_waveform(sample_rate, np.asarray(samples, dtype=np.float32))
            return self.identifier.compute(stream)
        except Exception as e:
            print(f"[Dictation] Language ID skipped: {e}")
            self.identifier = None
            return ""

    def _initialize(self):
        if not (self.encoder.exists() and self.decoder.exists()):
            return

        try:
            config = sherpa_onnx.SpokenLanguageIdentificationConfig(
                whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
                    encoder=str(self.encoder),
                    decoder=str(self.decoder),
                    tail_paddings=-1,
                ),
                num_threads=1,
                provider="cpu",
            )
            self.identifier = sherpa_onnx.SpokenLanguageIdentification(config)
        except Exception as e:
            print(f"[Dictation] Language ID disabled: {e}")
            self.identifier = None


class DictationPostProcessor:
    def __init__(self, rules_path: str = DICTATION_RULES_PATH):
        self.rules_path = rules_path
        self.profile = ""
        self.profiles = {}
        self.punctuation = PunctuationRestorer()
        self.llm = LLMTextRefiner()
        self.reload()

    def reload(self):
        self.filler_words = DEFAULT_FILLER_WORDS[:]
        self.custom_words = DEFAULT_CUSTOM_WORDS.copy()
        self.custom_phrases = DEFAULT_CUSTOM_PHRASES.copy()
        self.question_starters = DEFAULT_QUESTION_STARTERS[:]
        self.allowed_languages = DEFAULT_ALLOWED_LANGUAGES[:]
        self.fuzzy_threshold = 0.84
        self.profiles = {}
        rules = self._read_rules(self.rules_path)
        self._apply_rules(rules)
        self.profiles = rules.get("profiles", {}) if isinstance(rules.get("profiles"), dict) else {}
        self.llm = LLMTextRefiner(rules.get("llm", {}) if isinstance(rules.get("llm"), dict) else {})
        if self.profile:
            self.set_profile(self.profile)

    def set_profile(self, profile: str | None):
        self.profile = (profile or "").lower()
        rules = self._read_rules(self.rules_path)
        self.filler_words = DEFAULT_FILLER_WORDS[:]
        self.custom_words = DEFAULT_CUSTOM_WORDS.copy()
        self.custom_phrases = DEFAULT_CUSTOM_PHRASES.copy()
        self.question_starters = DEFAULT_QUESTION_STARTERS[:]
        self.allowed_languages = DEFAULT_ALLOWED_LANGUAGES[:]
        self.fuzzy_threshold = 0.84
        self._apply_rules(rules)

        profiles = rules.get("profiles", {}) if isinstance(rules.get("profiles"), dict) else {}
        profile_rules = profiles.get(self.profile, {})
        if isinstance(profile_rules, dict):
            self._apply_rules(profile_rules)

    def clean(self, text: str) -> str:
        text = self._normalize_spacing(text)
        text = self._apply_spoken_punctuation(text)
        text = self._remove_fillers(text)
        text = self._normalize_spacing(text)

        if not text:
            return ""

        text = self._apply_custom_phrases(text)
        text = self._apply_custom_words(text)
        if not re.search(r"[.!?]", text):
            text = self.punctuation.restore(text)
        text = self._normalize_spacing(text)
        text = re.sub(r"[,;:]+$", "", text).strip()

        if text[-1] not in ".!?":
            text += "?" if self._looks_like_question(text) else "."

        text = self._sentence_case(text)
        text = self._apply_custom_phrases(text)
        text = self._apply_custom_words(text)
        text = self.llm.refine(text)
        text = self._normalize_spacing(text)
        text = self._remove_fillers(text)
        text = self._apply_custom_phrases(text)
        text = self._apply_custom_words(text)
        text = self._sentence_case(text)
        text = self._apply_custom_phrases(text)
        text = self._apply_custom_words(text)
        return text

    @staticmethod
    def _read_rules(rules_path: str) -> dict:
        if not rules_path or not os.path.exists(rules_path):
            return {}

        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[Dictation] Ignoring rules file {rules_path}: {e}")
            return {}
        return data if isinstance(data, dict) else {}

    def _apply_rules(self, rules: dict):
        self.filler_words = self._merge_list(
            self.filler_words, rules.get("filler_words", [])
        )
        self.question_starters = self._merge_list(
            self.question_starters, rules.get("question_starters", [])
        )
        custom_words = rules.get("custom_words", {})
        if isinstance(custom_words, dict):
            for src, replacement in custom_words.items():
                self.custom_words[str(src)] = str(replacement)
        custom_phrases = rules.get("custom_phrases", {})
        if isinstance(custom_phrases, dict):
            for src, replacement in custom_phrases.items():
                self.custom_phrases[str(src)] = str(replacement)
        self.allowed_languages = self._merge_list(
            self.allowed_languages, rules.get("allowed_languages", [])
        )
        if "fuzzy_threshold" in rules:
            try:
                self.fuzzy_threshold = float(rules["fuzzy_threshold"])
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _merge_list(defaults, overrides):
        if not isinstance(overrides, list):
            return defaults
        merged = defaults[:]
        seen = {item.lower() for item in defaults}
        for item in overrides:
            item = str(item).strip()
            if item and item.lower() not in seen:
                merged.append(item)
                seen.add(item.lower())
        return merged

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])(?=[^\s,.!?;:])", r"\1 ", text)
        return text.strip()

    @staticmethod
    def _apply_spoken_punctuation(text: str) -> str:
        replacements = {
            "comma": ",",
            "period": ".",
            "full stop": ".",
            "question mark": "?",
            "exclamation mark": "!",
            "new line": "\n",
            "newline": "\n",
        }
        for phrase, punctuation in replacements.items():
            text = re.sub(
                rf"\b{re.escape(phrase)}\b",
                punctuation,
                text,
                flags=re.IGNORECASE,
            )
        return text

    def _remove_fillers(self, text: str) -> str:
        filler_pattern = "|".join(re.escape(word) for word in self.filler_words)
        if not filler_pattern:
            return text

        text = re.sub(
            rf"(?:^|(?<=[.!?]))\s*\b(?:{filler_pattern})\b[,.!?;:]*\s*",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(?:[\s,.!?;:]*\b(?:{filler_pattern})\b[\s,.!?;:]*)+$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    def _looks_like_question(self, text: str) -> bool:
        first_word = re.match(r"[A-Za-z']+", text.strip())
        return bool(first_word and first_word.group(0).lower() in self.question_starters)

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        chars = list(text)
        capitalize_next = True
        for i, char in enumerate(chars):
            if char.isalpha() and capitalize_next:
                chars[i] = char.upper()
                capitalize_next = False
            elif char in ".!?\n":
                capitalize_next = True
            elif not char.isspace():
                capitalize_next = False
        text = "".join(chars)
        return re.sub(r"\bi\b", "I", text)

    @classmethod
    def _sentence_case(cls, text: str) -> str:
        return cls._capitalize_sentences(text.lower())

    def _apply_custom_phrases(self, text: str) -> str:
        for src, replacement in sorted(
            self.custom_phrases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = re.sub(re.escape(src), replacement, text, flags=re.IGNORECASE)
        return text

    def _apply_custom_words(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z][A-Za-z']*|[^A-Za-z]+", text)
        corrected = []
        for token in tokens:
            if not re.match(r"[A-Za-z]", token):
                corrected.append(token)
                continue

            lower = token.lower()
            exact = self.custom_words.get(lower)
            if exact:
                corrected.append(exact)
                continue

            replacement = self._best_fuzzy_word(lower)
            corrected.append(replacement or token)

        return "".join(corrected)

    def _best_fuzzy_word(self, word: str) -> str | None:
        if len(word) < 3:
            return None

        best_score = 0.0
        best_replacement = None
        for src, replacement in self.custom_words.items():
            src = src.lower()
            if abs(len(src) - len(word)) > 2 or len(src) < 3:
                continue
            score = SequenceMatcher(None, word, src).ratio()
            if score > best_score:
                best_score = score
                best_replacement = replacement

        if best_score >= self.fuzzy_threshold:
            return best_replacement
        return None


# ---------------------------------------------------------------------------
# StreamingTranscriber  (Parakeet v3 - near-real-time dictation)
# ---------------------------------------------------------------------------
class StreamingTranscriber:
    def __init__(self, model_dir: str = MODEL_DIR_STREAMING):
        self.model_dir = model_dir
        self.recognizer = None
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self.on_text = None
        self.sample_rate = 16000
        self.vad_model_path = Path(VAD_MODEL_PATH)
        self.vad = None
        self.use_silero_vad = _truthy(os.environ.get("FINCH_USE_SILERO_VAD"), True)
        self.decode_on_silence = _truthy(os.environ.get("FINCH_DECODE_ON_SILENCE"), False)
        self.min_voice_rms = float(os.environ.get("FINCH_MIN_VOICE_RMS", "0.005"))
        self.noise_floor = float(os.environ.get("FINCH_NOISE_FLOOR_RMS", "0.0025"))
        self.vad_speech_factor = float(os.environ.get("FINCH_VAD_SPEECH_FACTOR", "2.0"))
        self.vad_silence_factor = float(os.environ.get("FINCH_VAD_SILENCE_FACTOR", "1.4"))
        self.pre_roll_s = 0.2
        self.start_speech_s = 0.12
        self.min_speech_s = 0.25
        self.trailing_silence_s = 0.55
        self.max_utterance_s = 12.0
        self.post_processor = DictationPostProcessor()
        self.language_validator = LanguageValidator()

    # ------------------------------------------------------------------
    def initialize(self):
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Dictation model not found: {self.model_dir}\n"
                "Run bin/download_models.sh to fetch models."
            )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=f"{self.model_dir}/tokens.txt",
            encoder=f"{self.model_dir}/encoder.int8.onnx",
            decoder=f"{self.model_dir}/decoder.int8.onnx",
            joiner=f"{self.model_dir}/joiner.int8.onnx",
            num_threads=4,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )
        self._initialize_vad()

    def _initialize_vad(self):
        if not (self.use_silero_vad and self.vad_model_path.exists()):
            return

        try:
            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(self.vad_model_path),
                    threshold=0.45,
                    min_silence_duration=self.trailing_silence_s,
                    min_speech_duration=self.min_speech_s,
                    max_speech_duration=self.max_utterance_s,
                    window_size=512,
                ),
                sample_rate=self.sample_rate,
                num_threads=1,
                provider="cpu",
            )
            self.vad = sherpa_onnx.VoiceActivityDetector(config, 60)
        except Exception as e:
            print(f"[Dictation] Silero VAD disabled: {e}")
            self.vad = None

    # ------------------------------------------------------------------
    def start(self, callback):
        self.on_text = callback
        if self.recognizer is None:
            self.initialize()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=2)

    def add_audio(self, samples):
        if self._running:
            self._q.put(np.asarray(samples, dtype=np.float32))

    def set_context(self, profile: str | None):
        self.post_processor.set_profile(profile)

    def reload_rules(self):
        self.post_processor.reload()

    # ------------------------------------------------------------------
    def _loop(self):
        if self.vad is not None:
            self._loop_silero_vad()
        else:
            self._loop_energy_vad()

    def _loop_silero_vad(self):
        self.vad.reset()
        utterance = []

        while True:
            try:
                samples = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if samples is None:
                self.vad.flush()
                self._drain_vad_segments(utterance)
                break

            self.vad.accept_waveform(np.asarray(samples, dtype=np.float32))
            self._drain_vad_segments(utterance)

            total_len = sum(len(chunk) for chunk in utterance)
            if total_len >= int(self.max_utterance_s * self.sample_rate):
                self._decode_chunks(utterance)
                utterance = []

        if utterance:
            self._decode_chunks(utterance)

    def _drain_vad_segments(self, utterance):
        while not self.vad.empty():
            segment = self.vad.front
            samples = self._segment_samples(segment)
            self.vad.pop()
            if len(samples) == 0:
                continue
            if self.decode_on_silence:
                self._decode_chunks([samples])
            else:
                utterance.append(samples)

    @staticmethod
    def _segment_samples(segment):
        samples = getattr(segment, "samples", None)
        if samples is None:
            return np.array([], dtype=np.float32)
        return np.asarray(samples, dtype=np.float32)

    def _loop_energy_vad(self):
        pre_roll = deque(maxlen=max(1, int(self.pre_roll_s * self.sample_rate / 1024)))
        chunks = []
        speech_samples = 0
        silence_samples = 0
        speech_frames = 0
        in_speech = False

        while True:
            try:
                samples = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if samples is None:
                break

            samples = np.asarray(samples, dtype=np.float32)
            is_voice = self._is_voice(samples, in_speech)

            if not in_speech:
                pre_roll.append(samples)
                if not is_voice:
                    speech_frames = 0
                    self._update_noise_floor(samples)
                    continue

                speech_frames += 1
                if speech_frames < self._speech_start_frames(len(samples)):
                    continue

                chunks = list(pre_roll)
                speech_samples = sum(len(chunk) for chunk in chunks)
                silence_samples = 0
                in_speech = True
                continue

            chunks.append(samples)

            if is_voice:
                speech_samples += len(samples)
                silence_samples = 0
            else:
                silence_samples += len(samples)
                self._update_noise_floor(samples)

            enough_speech = speech_samples >= int(self.min_speech_s * self.sample_rate)
            enough_silence = silence_samples >= int(self.trailing_silence_s * self.sample_rate)
            too_long = sum(len(chunk) for chunk in chunks) >= int(self.max_utterance_s * self.sample_rate)

            if enough_speech and (too_long or (self.decode_on_silence and enough_silence)):
                self._decode_chunks(chunks)
                pre_roll.clear()
                chunks = []
                speech_samples = 0
                silence_samples = 0
                speech_frames = 0
                in_speech = False

        if chunks:
            self._decode_chunks(chunks)

    def _decode_chunks(self, chunks):
        samples = np.concatenate(chunks)
        samples = self._trim_to_speech(samples)
        if len(samples) < int(self.min_speech_s * self.sample_rate):
            return
        if not self.language_validator.allowed(
            samples, self.sample_rate, self.post_processor.allowed_languages
        ):
            print("[Dictation] Skipped non-allowed language")
            return

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)
        text = self.post_processor.clean(stream.result.text)

        if text and self.on_text:
            self.on_text(text)
            print(f"[Dictation] {text}")

    def _is_voice(self, samples, in_speech: bool) -> bool:
        rms = self._rms(samples)
        factor = self.vad_silence_factor if in_speech else self.vad_speech_factor
        threshold = max(self.min_voice_rms, self.noise_floor * factor)
        return rms >= threshold

    def _update_noise_floor(self, samples):
        rms = self._rms(samples)
        if rms <= max(self.min_voice_rms, self.noise_floor * 3.0):
            self.noise_floor = (self.noise_floor * 0.98) + (max(rms, 0.0005) * 0.02)

    def _speech_start_frames(self, chunk_len: int) -> int:
        chunk_len = max(1, chunk_len)
        return max(1, int(self.start_speech_s * self.sample_rate / chunk_len))

    def _trim_to_speech(self, samples):
        frame_len = max(1, int(0.02 * self.sample_rate))
        if len(samples) < frame_len:
            return samples

        frame_count = len(samples) // frame_len
        frames = samples[: frame_count * frame_len].reshape(frame_count, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        low_rms = float(np.percentile(rms, 20))
        high_rms = float(np.percentile(rms, 80))
        threshold = max(self.min_voice_rms * 0.75, min(low_rms * 2.0, high_rms * 0.5))
        voiced = np.where(rms >= threshold)[0]
        if len(voiced) == 0:
            return np.array([], dtype=np.float32)

        pad = int(0.08 * self.sample_rate)
        start = max(0, int(voiced[0]) * frame_len - pad)
        end = min(len(samples), (int(voiced[-1]) + 1) * frame_len + pad)
        return samples[start:end]

    @staticmethod
    def _rms(samples) -> float:
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

    @staticmethod
    def _format_dictation_text(text: str) -> str:
        return DictationPostProcessor().clean(text)


# ---------------------------------------------------------------------------
# OfflineTranscriber  (Parakeet v3 – batch post-meeting)
# ---------------------------------------------------------------------------
class OfflineTranscriber:
    def __init__(self, model_dir: str = MODEL_DIR_OFFLINE):
        self.model_dir = model_dir
        self.recognizer = None

    # ------------------------------------------------------------------
    def initialize(self):
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Offline model not found: {self.model_dir}\n"
                "Run bin/download_models.sh to fetch models."
            )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=f"{self.model_dir}/tokens.txt",
            encoder=f"{self.model_dir}/encoder.int8.onnx",
            decoder=f"{self.model_dir}/decoder.int8.onnx",
            joiner=f"{self.model_dir}/joiner.int8.onnx",
            num_threads=4,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    # ------------------------------------------------------------------
    def transcribe_file(self, wav_path: str, on_text=None, on_done=None):
        """
        Transcribe a WAV file in a background thread.
        Calls on_text(segment_text) for each decoded chunk.
        Calls on_done(transcript_path) when complete.
        """
        t = threading.Thread(
            target=self._run, args=(wav_path, on_text, on_done), daemon=True
        )
        t.start()
        return t

    def _run(self, wav_path: str, on_text, on_done):
        if self.recognizer is None:
            self.initialize()

        print(f"[OfflineTranscriber] Processing {wav_path} …")
        samples, sr = self._load_wav(wav_path)

        # Chunk in 30-second windows to avoid empty-transcription on very long audio
        CHUNK_S = 30
        chunk_len = CHUNK_S * sr
        transcript_lines = []

        for i in range(0, len(samples), chunk_len):
            chunk = samples[i : i + chunk_len]
            if len(chunk) == 0:
                continue

            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, chunk.tolist())
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()

            if text:
                timestamp = self._fmt_ts(i / sr)
                line = f"[{timestamp}] {text}"
                transcript_lines.append(line)
                if on_text:
                    on_text(line)

        # Write transcript
        transcript_path = wav_path.replace(".wav", "_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

        print(f"[OfflineTranscriber] Done → {transcript_path}")
        if on_done:
            on_done(transcript_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_wav(path: str):
        """Load first channel, return (float32 numpy, sample_rate)."""
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        sampwidth = wf.getsampwidth() if hasattr(wf, "_sampwidth") else 2
        samples = np.frombuffer(raw, dtype=np.int16)

        # If stereo, take left channel (mic)
        if n_channels == 2:
            samples = samples[::2]

        return samples.astype(np.float32) / 32768.0, sr

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
