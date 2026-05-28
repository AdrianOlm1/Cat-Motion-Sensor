# Cat Motion Sensor

**A motion-detection security camera built with OpenCV that records clips and emails them to you when movement is detected.**

Originally built to catch a cat sneaking around — uses frame differencing to detect motion, automatically records video clips of the activity, and sends them as email attachments in a background thread.

## How It Works

1. **Frame Differencing** — Compares consecutive webcam frames to detect pixel changes above a threshold
2. **Motion Recording** — When motion is detected, starts recording an AVI clip using OpenCV's VideoWriter
3. **Email Alerts** — Sends the recorded clip as an email attachment via Gmail SMTP (runs in a separate thread to avoid blocking)

## Tech Stack

- **OpenCV** — Webcam capture, frame differencing, video recording
- **NumPy** — Frame comparison and threshold calculations
- **smtplib** — SMTP email delivery with attachments
- **threading** — Non-blocking email sending

## Setup

```bash
pip install opencv-python numpy

# Configure email in MotionSensor.py
# Set your Gmail address and app password
python MotionSensor/MotionSensor.py
```

> **Note:** Gmail requires an [App Password](https://support.google.com/accounts/answer/185833) for SMTP access.

## License

MIT
