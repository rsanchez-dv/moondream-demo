# moondream-demo

Real-time object detection using [Moondream](https://moondream.ai) (a Vision Language Model) with an **open-vocabulary** interface — detect anything by describing it in plain English, no retraining needed.

Supports:
- **USB / built-in webcam** via OpenCV
- **Tapo C100 IP camera** (or any RTSP stream) with threaded buffering to eliminate lag

---

## Project Structure

```
moondream-demo/
├── main.py        # Entry point — CLI args, main capture loop
├── detector.py    # Moondream model wrapper (cloud or local)
├── stream.py      # Video source abstractions (WebcamStream, TapoStream)
├── overlay.py     # Bounding box + HUD drawing helpers
├── config.py      # Defaults loaded from environment / .env
├── .env.example   # Copy to .env and fill in secrets
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note**: For local inference, a CUDA-capable GPU is strongly recommended for acceptable speed. CPU-only is possible but slow.

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
MOONDREAM_API_KEY=your_api_key_here     # from https://moondream.ai
TAPO_RTSP_URL=rtsp://user:pass@192.168.1.100:554/stream2   # optional
```

---

## Usage

### Cloud API + built-in webcam

```bash
python main.py --api-key YOUR_KEY
```

### Cloud API + Tapo C100 RTSP stream

```bash
python main.py --api-key YOUR_KEY \
               --source rtsp://user:pass@192.168.1.100:554/stream2 \
               --object person
```

### Local inference (no API key, no internet)

```bash
python main.py --local
```

### Detect something different

```bash
python main.py --api-key YOUR_KEY --object "coffee mug"
python main.py --api-key YOUR_KEY --object cat
python main.py --api-key YOUR_KEY --object car
```

### Tune performance (higher skip = faster UI, staler boxes)

```bash
python main.py --api-key YOUR_KEY --frame-skip 10
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Type a new detection label in the terminal without restarting |

---

## Tapo C100 RTSP Setup

1. Open the **Tapo app** → your camera → gear icon → **Device Settings**
2. Go to **Advanced Settings → Camera Account** → create a username + password
3. Note the camera IP under **Device Info**
4. Use `stream1` (1080p) or `stream2` (SD, less lag) in the RTSP URL

**Test in VLC first**: *Media → Open Network Stream → paste URL*

> ⚠️ Set a static IP for the camera (DHCP reservation in your router) so it doesn't change after reboots.

---

## How It Works

### Normalised coordinates
Moondream returns bounding boxes as values in `[0.0, 1.0]` relative to image dimensions. `overlay.py` converts them to pixel coords:
```python
x1_px = int(box.x_min * frame_width)
y1_px = int(box.y_min * frame_height)
```

### BGR → RGB conversion
OpenCV captures frames in **BGR**. Moondream (and PIL) expect **RGB**:
```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)
```

### Frame skipping
VLMs are significantly slower than traditional detectors (YOLO etc.). Frame skipping keeps the UI responsive while showing the last known boxes on intermediate frames:
```python
if frame_count % FRAME_SKIP == 0:
    boxes = detector.detect(pil_img, label)
```

### Threaded RTSP buffering
`TapoStream` runs a daemon thread that continuously calls `cap.grab()` to drain OpenCV's internal RTSP buffer. Without this, slow inference causes the buffer to fill with stale frames.

---

## Resources

- [Moondream Docs](https://docs.moondream.ai)
- [Moondream Playground](https://moondream.ai/c/playground)
- [GitHub: vikhyat/moondream](https://github.com/vikhyat/moondream)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Pillow Docs](https://pillow.readthedocs.io/en/stable/)
