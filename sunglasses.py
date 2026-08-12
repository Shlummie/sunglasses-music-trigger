import threading
import json
import os

import cv2
import numpy as np
import mediapipe as mp
import pygame
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvirtualcam
from pyvirtualcam import PixelFormat

SETTINGS_FILE = "sunglasses_settings.json"

# -------------- Audio setup --------------
pygame.mixer.init()


def start_music():
    pygame.mixer.music.stop()
    pygame.mixer.music.play()


def stop_music():
    pygame.mixer.music.stop()


# -------------- Mediapipe / eye region setup --------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDXS = [33, 133, 160, 158, 159, 144, 145, 153, 154, 155]
RIGHT_EYE_IDXS = [362, 263, 387, 385, 386, 373, 374, 380, 381, 382]


def analyze_eye(frame, face_landmarks):
    """
    Returns: (mean_brightness, texture_value)
    texture_value = variance of Laplacian (edge detail).
    No drawing/overlay is done on the frame.
    """
    h, w, _ = frame.shape
    points = []

    for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = max(min(xs) - 5, 0), min(max(xs) + 5, w)
    y_min, y_max = max(min(ys) - 5, 0), min(max(ys) + 5, h)

    eye_roi = frame[y_min:y_max, x_min:x_max]
    if eye_roi.size == 0:
        return None, None

    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = lap.var()

    return mean_brightness, texture


# -------------- GUI app --------------
class SunglassesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sunglasses MP3 Player")

        # GUI state vars
        self.mp3_path = tk.StringVar(value="song.mp3")
        self.status_var = tk.StringVar(value="Status: Idle")
        self.sunglasses_var = tk.StringVar(value="Sunglasses: OFF")
        self.brightness_var = tk.StringVar(value="Eye brightness: N/A")
        self.texture_var = tk.StringVar(value="Texture: N/A")

        self.running = False
        self.thread = None

        # Detection tuning (play when texture > threshold)
        self.texture_threshold_value = 150.0

        # Audio volume (0.0–1.0)
        self.volume_value = 0.8

        # Shared state for UI updates
        self._wearing_sunglasses = False
        self._status_msg = "Status: Idle"
        self._last_brightness = None
        self._last_texture = None

        # Load settings from previous session (if any)
        self.load_settings()

        pygame.mixer.music.set_volume(self.volume_value)

        # Entry text for threshold, based on possibly loaded settings
        self.texture_threshold_str = tk.StringVar(
            value=str(self.texture_threshold_value)
        )

        self.build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.poll_state)

    # ---------- Settings persistence ----------
    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Could not load settings:", e)
            return

        mp3 = data.get("mp3_path")
        if isinstance(mp3, str) and mp3:
            self.mp3_path.set(mp3)

        tex = data.get("texture_threshold")
        if isinstance(tex, (int, float)):
            self.texture_threshold_value = float(tex)

        vol = data.get("volume")
        if isinstance(vol, (int, float)):
            self.volume_value = max(0.0, min(1.0, float(vol)))

    def save_settings(self):
        # Make sure threshold is a valid number first
        if not self._update_threshold_from_entry():
            return

        data = {
            "mp3_path": self.mp3_path.get(),
            "texture_threshold": self.texture_threshold_value,
            "volume": self.volume_value,
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings:\n{e}")
            return

        messagebox.showinfo("Saved", "Settings saved.")

    # ---------- UI build ----------
    def build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        # MP3 picker
        ttk.Label(frm, text="MP3 file:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(frm, textvariable=self.mp3_path, width=40)
        entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(frm, text="Browse", command=self.browse_mp3).grid(
            row=0, column=2, padx=(5, 0)
        )

        # Texture threshold entry
        ttk.Label(
            frm, text="Texture threshold (play when texture > value):"
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Label(frm, text="Value:").grid(row=2, column=0, sticky="w")
        self.texture_entry = ttk.Entry(frm, textvariable=self.texture_threshold_str)
        self.texture_entry.grid(row=2, column=1, columnspan=2, sticky="ew")

        # Volume slider
        ttk.Label(frm, text="Volume:").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        self.volume_scale = ttk.Scale(
            frm,
            from_=0,
            to=100,
            value=self.volume_value * 100,
            command=self.on_volume_change,
        )
        self.volume_scale.grid(row=4, column=0, columnspan=3, sticky="ew")

        # Status labels
        ttk.Label(frm, textvariable=self.status_var).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        ttk.Label(
            frm,
            textvariable=self.sunglasses_var,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 0))

        ttk.Label(frm, textvariable=self.brightness_var).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(5, 0)
        )
        ttk.Label(frm, textvariable=self.texture_var).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=(5, 0))

        self.start_btn = ttk.Button(
            btn_frame, text="Start detection", command=self.start_detection
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self.stop_detection, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="Save settings", command=self.save_settings).pack(
            side="left", padx=5
        )

        ttk.Button(btn_frame, text="Quit", command=self.on_close).pack(
            side="left", padx=5
        )

        frm.columnconfigure(1, weight=1)

    # ---------- UI callbacks ----------
    def browse_mp3(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")],
        )
        if path:
            self.mp3_path.set(path)

    def on_volume_change(self, value):
        try:
            volume_percent = float(value)
        except ValueError:
            return
        self.volume_value = max(0.0, min(1.0, volume_percent / 100.0))
        pygame.mixer.music.set_volume(self.volume_value)

    def _update_threshold_from_entry(self):
        text = self.texture_threshold_str.get().strip()
        if not text:
            messagebox.showerror(
                "Error", "Texture threshold cannot be empty. Enter a number."
            )
            self.texture_threshold_str.set(str(self.texture_threshold_value))
            return False
        try:
            self.texture_threshold_value = float(text)
        except ValueError:
            messagebox.showerror(
                "Error", "Texture threshold must be a number (e.g. 150.0)."
            )
            self.texture_threshold_str.set(str(self.texture_threshold_value))
            return False
        return True

    def start_detection(self):
        if self.running:
            return

        # Make sure threshold is valid before starting
        if not self._update_threshold_from_entry():
            return

        try:
            pygame.mixer.music.load(self.mp3_path.get())
        except Exception as e:
            messagebox.showerror("Error", f"Could not load audio file:\n{e}")
            return

        pygame.mixer.music.set_volume(self.volume_value)

        self.running = True
        self._status_msg = "Status: Detecting"
        self._wearing_sunglasses = False
        self._last_brightness = None
        self._last_texture = None

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def stop_detection(self):
        self.running = False
        stop_music()

    # ---------- Camera / detection thread ----------
    def capture_loop(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self._status_msg = "Status: Could not open webcam"
            self.running = False
            return

        # Get frame size for virtual camera
        ret, frame = cap.read()
        if not ret:
            self._status_msg = "Status: Camera error"
            cap.release()
            self.running = False
            return

        h, w = frame.shape[:2]

        STABLE_FRAMES = 5
        frames_with_glasses = 0
        frames_without_glasses = 0
        wearing_sunglasses = False

        with pyvirtualcam.Camera(
            width=w, height=h, fps=30, fmt=PixelFormat.BGR
        ) as vcam:
            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as face_mesh:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        self._status_msg = "Status: Camera error"
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    sunglasses_now = False
                    mean_brightness = None
                    texture = None

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        mean_brightness, texture = analyze_eye(frame, face_landmarks)

                        # MAIN RULE: play when texture > threshold
                        if texture is not None:
                            sunglasses_now = texture > self.texture_threshold_value
                        else:
                            sunglasses_now = False
                    else:
                        sunglasses_now = False

                    self._last_brightness = mean_brightness
                    self._last_texture = texture

                    if sunglasses_now:
                        frames_with_glasses += 1
                        frames_without_glasses = 0
                    else:
                        frames_without_glasses += 1
                        frames_with_glasses = 0

                    if not wearing_sunglasses and frames_with_glasses >= STABLE_FRAMES:
                        wearing_sunglasses = True
                        self._wearing_sunglasses = True
                        start_music()

                    if wearing_sunglasses and frames_without_glasses >= STABLE_FRAMES:
                        wearing_sunglasses = False
                        self._wearing_sunglasses = False
                        stop_music()

                    self._status_msg = "Status: Detecting"

                    # Send clean frame (no overlays) to virtual webcam
                    vcam.send(frame)
                    vcam.sleep_until_next_frame()

        cap.release()
        stop_music()
        self.running = False
        self._status_msg = "Status: Idle"
        self._wearing_sunglasses = False

    # ---------- GUI polling / shutdown ----------
    def poll_state(self):
        self.status_var.set(self._status_msg)
        self.sunglasses_var.set(
            "Sunglasses: ON" if self._wearing_sunglasses else "Sunglasses: OFF"
        )

        if self._last_brightness is None:
            self.brightness_var.set("Eye brightness: N/A")
        else:
            self.brightness_var.set(f"Eye brightness: {int(self._last_brightness)}")

        if self._last_texture is None:
            self.texture_var.set("Texture: N/A")
        else:
            self.texture_var.set(f"Texture: {int(self._last_texture)}")

        if self.running:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

        self.root.after(200, self.poll_state)

    def on_close(self):
        self.running = False
        stop_music()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        pygame.mixer.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SunglassesApp(root)
    root.mainloop()
