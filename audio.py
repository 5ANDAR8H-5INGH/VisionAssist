"""Non-blocking offline TTS with per-message cooldown."""
import queue
import threading
import time


class Speaker:
    def __init__(self, enabled=True, cooldown=3.0):
        self.enabled = enabled
        self.cooldown = cooldown
        self.q = queue.Queue(maxsize=1)
        self.last = {}
        self._stop = False
        if enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:                      # no TTS backend available
            print(f"[audio] TTS unavailable: {e}")
            self.enabled = False
            return
        while not self._stop:
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            engine.say(text)
            engine.runAndWait()

    def say(self, text, key):
        if not self.enabled:
            return
        now = time.time()
        if now - self.last.get(key, 0) < self.cooldown:
            return
        self.last[key] = now
        try:                                        # drop stale message
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def close(self):
        self._stop = True
