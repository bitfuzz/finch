"""
main.py
Finch Transcriber entry point.
 - System tray icon (pystray)
 - Hotkeys: hold Ctrl+Space (dictation), Ctrl+Alt+R (meeting)
 - Dictation:  Parakeet v3 chunks -> keyboard.write()
 - Meeting:    Raw stereo WAV → Parakeet v3 batch after stop
"""
from __future__ import annotations
import os
import ctypes
import threading
import pynput
import pystray
from PIL import Image, ImageDraw

from audio_capture   import AudioCapture
from transcriber     import StreamingTranscriber, OfflineTranscriber, DICTATION_RULES_PATH
from dictation_ui    import DictationUI, run_ui_threaded
from system_audio    import SystemAudioMuter


# ---------------------------------------------------------------------------
class App:
    def __init__(self):
        self.audio       = AudioCapture()
        self.streaming   = StreamingTranscriber()
        self.offline     = OfflineTranscriber()
        self.ui          = DictationUI()
        self.system_audio = SystemAudioMuter()

        self.mode        = None   # "dictation" | "meeting" | None
        self.icon        = None
        self._dictation_hotkey_down = False

    # ------------------------------------------------------------------
    # Dictation
    # ------------------------------------------------------------------
    def toggle_dictation(self):
        if self.mode == "meeting":
            return

        if self.mode == "dictation":
            self._stop_dictation()
        else:
            self._start_dictation()

    def _on_dictation_press(self):
        if self._dictation_hotkey_down or self.mode == "meeting":
            return
        self._dictation_hotkey_down = True
        if self.mode != "dictation":
            self._start_dictation()

    def _on_dictation_release(self, event=None):
        if not self._dictation_hotkey_down:
            return
        self._dictation_hotkey_down = False
        if self.mode == "dictation":
            self._stop_dictation()

    def _start_dictation(self):
        self.mode = "dictation"
        self.streaming.set_context(self._foreground_process_name())
        self.system_audio.mute()
        self.ui.show()
        self.streaming.start(self._on_dictation_text)
        self.audio.start_dictation(
            data_callback=self.streaming.add_audio,
            vis_callback=self.ui.update_visualizer,
        )
        print("[Finch] Dictation started")

    def _stop_dictation(self):
        self.audio.stop()
        self.system_audio.restore()
        self.streaming.stop()
        self.ui.hide()
        self.mode = None
        print("[Finch] Dictation stopped")

    def _on_dictation_text(self, text: str):
        controller = pynput.keyboard.Controller()
        controller.type(text + " ")
        print(f"  Dictated: {text}")

    # ------------------------------------------------------------------
    # Meeting
    # ------------------------------------------------------------------
    def toggle_meeting(self):
        if self.mode == "dictation":
            return

        if self.mode == "meeting":
            self._stop_meeting()
        else:
            self._start_meeting()

    def _start_meeting(self):
        self.mode = "meeting"
        self.audio.start_meeting()
        print("[Finch] Meeting recording started")

    def _stop_meeting(self):
        wav_path = self.audio.stop()
        self.mode = None
        print("[Finch] Meeting recording stopped")
        if wav_path:
            self._transcribe_meeting_async(wav_path)

    def _transcribe_meeting_async(self, wav_path: str):
        print(f"[Finch] Transcribing {wav_path} in background …")
        self.offline.transcribe_file(
            wav_path,
            on_text=lambda line: print(f"  {line}"),
            on_done=lambda path: print(f"[Finch] Transcript saved → {path}"),
        )

    def _reload_dictation_rules(self):
        self.streaming.reload_rules()
        print("[Finch] Dictation rules reloaded")

    def _open_dictation_rules(self):
        try:
            os.startfile(os.path.abspath(DICTATION_RULES_PATH))
        except OSError as e:
            print(f"[Finch] Cannot open dictation rules: {e}")

    # ------------------------------------------------------------------
    # Tray / Lifecycle
    # ------------------------------------------------------------------
    def _make_icon_image(self) -> Image.Image:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([8, 8, 56, 56], fill=(80, 200, 120))   # green circle
        return img

    def _quit(self):
        self.audio.stop()
        self.streaming.stop()
        self.system_audio.restore()
        if self.icon:
            self.icon.stop()
        os._exit(0)

    @staticmethod
    def _foreground_process_name() -> str:
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = kernel32.OpenProcess(0x1000, False, pid.value)
            if not process:
                return ""
            try:
                buffer = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(len(buffer))
                if kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
                    return os.path.basename(buffer.value).lower()
            finally:
                kernel32.CloseHandle(process)
        except Exception:
            return ""
        return ""

    def run(self):
        # Dictation UI (tkinter) must run on its own thread
        if os.environ.get("FINCH_NO_UI") == "1":
            print("[Finch] Running without Dictation UI (FINCH_NO_UI=1).")
        else:
            run_ui_threaded(self.ui)

        # Hotkeys via pynput
        self._ctrl_down = False
        self._shift_down = False

        def on_press(key):
            if key in (pynput.keyboard.Key.ctrl, pynput.keyboard.Key.ctrl_l, pynput.keyboard.Key.ctrl_r):
                self._ctrl_down = True
            elif key in (pynput.keyboard.Key.shift, pynput.keyboard.Key.shift_l, pynput.keyboard.Key.shift_r):
                self._shift_down = True
            elif key == pynput.keyboard.Key.space:
                if self._ctrl_down:
                    self._on_dictation_press()
            elif hasattr(key, 'char') and key.char and key.char.lower() == 'r':
                if self._ctrl_down and self._shift_down:
                    self.toggle_meeting()

        def on_release(key):
            if key in (pynput.keyboard.Key.ctrl, pynput.keyboard.Key.ctrl_l, pynput.keyboard.Key.ctrl_r):
                self._ctrl_down = False
            elif key in (pynput.keyboard.Key.shift, pynput.keyboard.Key.shift_l, pynput.keyboard.Key.shift_r):
                self._shift_down = False
            elif key == pynput.keyboard.Key.space:
                self._on_dictation_release()

        listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # System tray
        menu = pystray.Menu(
            pystray.MenuItem("Dictation  (hold Ctrl+Space)", lambda: self.toggle_dictation()),
            pystray.MenuItem("Meeting    (Ctrl+Shift+R)", lambda: self.toggle_meeting()),
            pystray.MenuItem("Reload Dictation Rules",  lambda: self._reload_dictation_rules()),
            pystray.MenuItem("Open Dictation Rules",    lambda: self._open_dictation_rules()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit",                    lambda: self._quit()),
        )
        self.icon = pystray.Icon(
            "Finch", self._make_icon_image(), "Finch Transcriber", menu
        )
        print("[Finch] Ready — hold Ctrl+Space: dictation | Ctrl+Shift+R: meeting")
        self.icon.run()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    App().run()
