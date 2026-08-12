# Sunglasses Music Trigger

A local webcam experiment that starts a user-selected audio track when the eye region appears to contain the extra edge texture associated with sunglasses. It also mirrors the clean camera frame to an optional virtual camera.

## What it does

- Captures webcam frames with OpenCV.
- Detects facial landmarks with MediaPipe Face Mesh.
- Crops a combined eye region from fixed landmark indices.
- Computes brightness and Laplacian variance as a simple texture signal.
- Applies a configurable threshold and stability logic before changing state.
- Starts or stops an MP3 through Pygame.
- Sends unannotated frames to `pyvirtualcam` for use in compatible conferencing or streaming software.

This is a heuristic demonstration, not a trained sunglasses classifier. Eyeglass frames, hair, shadows, motion blur, camera sharpening, and lighting can all change the texture score.

## Install

Requirements: Python 3.10+, a webcam, an audio output device, and an installed virtual-camera backend if virtual output is desired.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python sunglasses.py
```

Choose an MP3 you are licensed to play, tune the texture threshold, and save the local settings. `sunglasses_settings.example.json` documents the format; the actual `sunglasses_settings.json` is ignored because it can contain an absolute personal file path.

## Data flow

```text
webcam -> face landmarks -> eye crop -> Laplacian variance -> stable on/off state
       -> clean frame ---------------------------------------> virtual camera
                                                      +------> local MP3 playback
```

## Privacy and safety

Frames are processed locally and are not intentionally saved or uploaded. A virtual camera makes frames available to other local applications, so verify which application is consuming it. No music file is distributed with this repository.

The detector can be wrong and should not be used for identity, safety, access control, or surveillance decisions.

## License

Original source is MIT licensed. MediaPipe, OpenCV, NumPy, Pygame, pyvirtualcam, and any selected audio retain their own licenses.
