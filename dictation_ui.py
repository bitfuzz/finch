"""
dictation_ui.py
Borderless floating visualizer for dictation mode.
Runs in its own thread (tkinter mainloop).
"""
import threading
import threading


class DictationUI:
    def __init__(self):
        self.root    = None
        self.canvas  = None
        self._bar    = None
        self._visible = False
        self._ready   = threading.Event()

    # ------------------------------------------------------------------
    def run(self):
        """Call once in a dedicated thread."""
        import tkinter as tk
        self.root = tk.Tk()
        self.root.overrideredirect(True)    # no title bar
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)
        self.root.configure(bg="#0d0d0d")

        W, H = 180, 24
        # Position: bottom-right of screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{sw - W - 20}+{sh - H - 60}")

        self.canvas = tk.Canvas(
            self.root, width=W, height=H,
            bg="#0d0d0d", highlightthickness=0
        )
        self.canvas.pack()

        # Background track
        self.canvas.create_rectangle(0, 0, W, H, fill="#1a1a1a", outline="")
        # Active bar
        self._bar = self.canvas.create_rectangle(
            0, 4, 0, H - 4, fill="#50c878", outline=""
        )
        # Microphone label
        self.canvas.create_text(
            W - 6, H // 2, text="🎙", anchor="e",
            fill="#50c878", font=("Helvetica", 10)
        )

        self.root.withdraw()   # hidden until show() called
        self._ready.set()
        self.root.mainloop()

    # ------------------------------------------------------------------
    def update_visualizer(self, rms: float):
        if not self._visible or self.canvas is None:
            return
        width = min(max(int(rms * 900), 4), 160)
        self.root.after(0, lambda: self.canvas.coords(self._bar, 0, 4, width, 20))

    def show(self):
        if self.root is None:
            return
        self._ready.wait()
        self._visible = True
        self.root.after(0, self.root.deiconify)

    def hide(self):
        if self.root is None:
            return
        self._visible = False
        if self.root:
            self.root.after(0, self.root.withdraw)


# ---------------------------------------------------------------------------
def run_ui_threaded(ui: DictationUI) -> threading.Thread:
    t = threading.Thread(target=ui.run, daemon=True)
    t.start()
    ui._ready.wait()   # block until tkinter is ready
    return t
