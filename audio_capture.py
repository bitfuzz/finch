"""
audio_capture.py
Mic + system audio capture.
 - Dictation mode: mic only → float32 callbacks
 - Meeting mode:   stereo WAV writer (mic L, system R) → file on disk
"""
from __future__ import annotations
import os
import queue
import threading
import wave
import datetime

import numpy as np
import pyaudio
import soundcard as sc

import warnings
warnings.filterwarnings("ignore", message=".*loopback recording.*")

SAMPLE_RATE = 16000
CHUNK = 1024  # frames per buffer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _session_filename() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("recordings", exist_ok=True)
    return os.path.join("recordings", f"meeting_{ts}.wav")


# ---------------------------------------------------------------------------
class AudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk: int = CHUNK):
        self.sample_rate = sample_rate
        self.chunk = chunk

        self._pa = pyaudio.PyAudio()
        self._mic_stream = None
        self._loopback_thread: threading.Thread | None = None

        # Callbacks for dictation
        self._data_cb = None   # f(np.ndarray)
        self._vis_cb  = None   # f(float rms)

        # Meeting WAV writer state
        self._wav_file: wave.Wave_write | None = None
        self._wav_path: str | None = None
        self._wav_lock = threading.Lock()
        self._running = False
        self.mic_muted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_dictation(self, data_callback, vis_callback=None):
        """Mic-only capture → data_callback(np.ndarray), vis_callback(float)."""
        self._data_cb = data_callback
        self._vis_cb  = vis_callback
        self._running = True
        self._open_mic(callback=self._dictation_chunk)

    def start_meeting(self) -> str:
        """
        Stereo WAV capture: mic → left channel, system audio → right channel.
        Returns the path to the WAV file being written.
        """
        self._running = True
        self._wav_path = _session_filename()
        self._wav_file = wave.open(self._wav_path, "wb")
        self._wav_file.setnchannels(2)
        self._wav_file.setsampwidth(2)   # int16
        self._wav_file.setframerate(self.sample_rate)

        # Open mic stream (writes to WAV via callback)
        self._open_mic(callback=self._meeting_chunk)

        # Start system audio capture thread (soundcard loopback)
        self._loopback_thread = threading.Thread(
            target=self._loopback_loop, daemon=True
        )
        self._loopback_thread.start()

        print(f"[AudioCapture] Recording → {self._wav_path}")
        return self._wav_path

    def stop(self) -> str | None:
        """Stop all capture. Returns WAV path if in meeting mode, else None."""
        self._running = False

        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None

        if self._loopback_thread:
            self._loopback_thread.join(timeout=3)
            self._loopback_thread = None

        with self._wav_lock:
            if self._wav_file:
                self._wav_file.close()
                path = self._wav_path
                self._wav_file = None
                self._wav_path = None
                print(f"[AudioCapture] File closed → {path}")
                return path

        return None

    # ------------------------------------------------------------------
    # Internal – Mic stream
    # ------------------------------------------------------------------
    def _open_mic(self, callback):
        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=callback,
        )
        self._mic_stream.start_stream()

    # ------------------------------------------------------------------
    # Dictation chunk callback (PyAudio thread)
    # ------------------------------------------------------------------
    def _dictation_chunk(self, in_data, frame_count, time_info, status):
        samples_int = np.frombuffer(in_data, dtype=np.int16)
        samples_f32 = samples_int.astype(np.float32) / 32768.0

        if self._data_cb:
            self._data_cb(samples_f32)

        if self._vis_cb:
            rms = float(np.sqrt(np.mean(samples_f32 ** 2)))
            self._vis_cb(rms)

        return (None, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Meeting chunk callback (PyAudio thread) – mic → left channel
    # ------------------------------------------------------------------
    _sys_buffer: queue.Queue = queue.Queue(maxsize=256)  # class-level ring

    def _meeting_chunk(self, in_data, frame_count, time_info, status):
        mic_int = np.frombuffer(in_data, dtype=np.int16)
        if self.mic_muted:
            mic_int = np.zeros_like(mic_int)

        # Pull matching system audio frame (or silence if unavailable)
        try:
            sys_int = self._sys_buffer.get_nowait()
        except queue.Empty:
            sys_int = np.zeros(len(mic_int), dtype=np.int16)

        # Interleave L(mic) R(system) into stereo int16
        stereo = np.empty(len(mic_int) * 2, dtype=np.int16)
        stereo[0::2] = mic_int
        stereo[1::2] = sys_int[: len(mic_int)]

        with self._wav_lock:
            if self._wav_file:
                self._wav_file.writeframes(stereo.tobytes())

        return (None, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # System audio loopback (soundcard) – runs in thread
    # ------------------------------------------------------------------
    def _loopback_loop(self):
        import sys
        mic = None
        if sys.platform == "darwin":
            # macOS doesn't support native loopback. Find BlackHole virtual mic.
            for m in sc.all_microphones():
                if "BlackHole" in m.name:
                    mic = m
                    break
            if not mic:
                print("\n[AudioCapture] 'BlackHole' audio driver not found. Cannot capture system audio on macOS.")
                print(" -> Run: brew install blackhole-2ch\n")
                return
        else:
            try:
                # Windows WASAPI loopback
                mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
            except Exception:
                try:
                    mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)
                except Exception as e:
                    print(f"\n[AudioCapture] System audio unavailable: {e}\n")
                    return

        with mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
            while self._running:
                data = rec.record(numframes=self.chunk)
                samples_int = (data[:, 0] * 32767).astype(np.int16)
                try:
                    self._sys_buffer.put_nowait(samples_int)
                except queue.Full:
                    pass  # drop oldest – real-time priority
