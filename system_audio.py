"""
system_audio.py
Optional system output mute for dictation.
"""
from __future__ import annotations
import os


def _enabled(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SystemAudioMuter:
    def __init__(self, enabled: bool | None = None):
        self.enabled = _enabled(os.environ.get("FINCH_MUTE_SYSTEM_AUDIO"), True)
        if enabled is not None:
            self.enabled = enabled
        self._volume = None
        self._was_muted = None

    def mute(self):
        if not self.enabled:
            return
        volume = self._get_volume()
        if not volume:
            return
        try:
            self._was_muted = bool(volume.GetMute())
            volume.SetMute(1, None)
        except Exception as e:
            print(f"[Audio] System mute unavailable: {e}")

    def restore(self):
        if not self.enabled or self._was_muted is None:
            return
        volume = self._get_volume()
        if not volume:
            return
        try:
            if not self._was_muted:
                volume.SetMute(0, None)
        except Exception as e:
            print(f"[Audio] System unmute unavailable: {e}")
        finally:
            self._was_muted = None

    def _get_volume(self):
        if self._volume:
            return self._volume
        try:
            from ctypes import POINTER, cast
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            device = AudioUtilities.GetSpeakers()
            interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            self._volume = None
        return self._volume
