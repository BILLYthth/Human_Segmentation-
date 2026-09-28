# Real-Time Face Segmentation on Raspberry Pi (U-Net)

A U-Net-based face segmentation pipeline trained from a self-collected dataset, optimized and deployed for real-time inference on Raspberry Pi.

**Course:** Embedded Systems Lab — HCMUTE, Faculty of International Education
**Lecturer:** Assoc. Prof. PhD. Vo Minh Huan
**Team:** Nguyen Trong Duc, Nguyen Dang Truong Giang, Vu Thanh Dat

## Overview

This project builds a full edge-AI pipeline for face segmentation: from custom data collection, through U-Net training and evaluation, to model compression (quantization + pruning) and real-time deployment on Raspberry Pi with MQTT integration.

## Pipeline

1. **Dataset** — 6,000 images sourced from Kaggle, each paired with a segmentation mask. Split 70/15/15 (train/val/test).
2. **Model Training** — U-Net trained for face/body segmentation, evaluated with IoU and Dice Score.
3. **Quantization** — INT8 quantization for size/latency reduction, benchmarked against FP32 on Raspberry Pi.
4. **Pruning** — Structured filter pruning (0.5x, 0.75x) combined with quantization for further compression.
5. **Parallel Inference** — Multithreading and multiprocessing pipelines to improve real-time FPS on Raspberry Pi's limited CPU.
6. **Edge Deployment** — Real-time inference pipeline publishing face/mask metrics (`has_face`, `face_count`, `mask_area_total`, `fps`) over MQTT.

## Key Results

| Stage | Metric | Result |
|---|---|---|
| Baseline U-Net (300 epochs) | Mean IoU / Dice | 0.7056 / 0.7935 |
| Baseline U-Net | Val Accuracy / Loss | 0.9322 / 0.2264 |
| FP32 → INT8 (Raspberry Pi) | FPS | 1.65 → 4.71 |
| FP32 → INT8 | Model size | 23.5 MB → 6.4 MB |
| Pruned 0.5x + INT8 | Model size | 2.0 MB |
| Single → Multiprocessing | FPS | 4.70 → 4.78 |

## Repository Structure

```
├── src/                 # Training, quantization, pruning, deployment scripts
├── models/              # Trained U-Net weights (FP32 / INT8, baseline / pruned)
├── reports/             # Lab reports (W1–W7)
└── README.md
```

## Tech Stack

- **Modeling:** TensorFlow / Keras (U-Net), TensorFlow Lite (quantization)
- **Deployment:** Raspberry Pi, OpenCV, Python multiprocessing/multithreading, MQTT
- **Evaluation:** IoU, Dice Score, FPS, CPU usage, latency benchmarking

## Notes

- Trained model shows a color bias toward blue clothing (from training data distribution) — a known limitation discussed in the reports.
- Full lab progression (W1–W7) documents data collection, training, and each optimization/deployment step in detail.

## License

MIT

