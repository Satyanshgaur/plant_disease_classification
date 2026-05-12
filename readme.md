# 🌿 Plant Disease Classifier & Autonomous Field Scout

> An end-to-end machine learning and robotics project for early plant disease detection in agricultural fields using computer vision and autonomous navigation.

---

## Table of Contents

- [Overview](#overview)
- [Vision](#vision)
- [System Architecture](#system-architecture)
- [Features](#features)
  - [Machine Learning](#machine-learning)
  - [Robotics](#robotics)
  - [Data & Infrastructure](#data--infrastructure)
- [Pipeline](#pipeline)
- [Roadmap](#roadmap)
- [Use Cases](#use-cases)
- [Contributing](#contributing)

---

## Overview

This project aims to build an intelligent field monitoring system capable of:

- 🔬 Detecting diseased plant leaves using deep learning
- 🏷️ Identifying disease categories from live camera input
- 🤖 Operating autonomously on a wheeled robotic platform
- 🗺️ Scanning large agricultural areas with minimal human intervention
- 🌾 Assisting farmers with early disease detection and crop monitoring

The long-term goal is to combine **computer vision**, **embedded systems**, and **robotics** into a fully deployable precision agriculture system.

---

## Vision

The final system is planned to consist of:

| Component | Description |
|-----------|-------------|
| Mobile Robot Platform | Wheeled rover capable of navigating crop field terrain |
| Mounted Camera | Continuously scanning plants and leaves during traversal |
| Onboard ML Model | Real-time disease classification at the edge |
| GPS / Path Planning | Structured field traversal with coverage mapping *(optional)* |
| Alert Dashboard | Farmer-facing interface for detected diseases and locations |

---

## System Architecture

```
Camera Feed
     │
     ▼
Disease Detection Model
     │
     ▼
Prediction / Localization
     │
     ▼
Farmer Alert  ──or──  Autonomous Action
```

The pipeline is designed to run end-to-end on embedded hardware, with optional offloading to a cloud backend for logging, model updates, and dashboard access.

---

## Features

### Machine Learning

- **Leaf Disease Image Classification** — Multi-class identification of common crop diseases from leaf imagery
- **Transfer Learning with CNN Architectures** — Fine-tuned models (e.g., ResNet, EfficientNet, MobileNet) for high accuracy with limited data
- **Real-Time Inference Optimization** — Model quantization and pruning for edge deployment
- **Multi-Class Disease Detection** — Simultaneous identification of multiple disease types
- **Confidence Scoring & Visualization** — Per-prediction confidence scores with annotated output overlays

### Robotics

- **Wheeled Autonomous Rover** — Ground vehicle designed for row-crop navigation
- **Camera Stabilization / Mount System** — Dampened or gimbal-mounted camera for consistent image quality in motion
- **Obstacle Avoidance** — Sensor-based detection and re-routing around field obstacles
- **Field Navigation Logic** — Row-following and coverage algorithms for systematic scanning
- **Edge Deployment on Embedded Hardware** — Runs on platforms such as Raspberry Pi, Jetson Nano, or similar SBCs

### Data & Infrastructure

- **Dataset Collection & Preprocessing Pipeline** — Tooling for image ingestion, labeling, augmentation, and splitting
- **Model Training / Evaluation Framework** — Reproducible training scripts with validation metrics (accuracy, F1, confusion matrix)
- **Experiment Tracking** — Integrated logging (e.g., MLflow, Weights & Biases) for comparing runs
- **Model Versioning** — Checkpointing and versioned artifact storage
- **Inference Benchmarking** — Latency and throughput profiling across target hardware targets

---

## Pipeline

```
Raw Field Images
       │
       ▼
  Preprocessing
  (resize, normalize, augment)
       │
       ▼
  Model Training
  (transfer learning + fine-tuning)
       │
       ▼
  Evaluation & Validation
  (accuracy, F1, confusion matrix)
       │
       ▼
  Edge Optimization
  (quantization, pruning, ONNX export)
       │
       ▼
  Deployment on Robot
  (embedded inference + camera integration)
       │
       ▼
  Field Operation
  (autonomous navigation + real-time detection)
       │
       ▼
  Farmer Dashboard / Alerts
```

---

## Roadmap

- [ ] Dataset curation and preprocessing pipeline
- [ ] Baseline CNN classifier (transfer learning)
- [ ] Model evaluation framework and benchmarks
- [ ] Edge optimization and ONNX/TFLite export
- [ ] Rover hardware assembly and motor control
- [ ] Camera integration and image capture pipeline
- [ ] Real-time inference on embedded hardware
- [ ] Basic row-following navigation logic
- [ ] Obstacle avoidance with ultrasonic/LiDAR sensors
- [ ] GPS integration for field mapping *(stretch goal)*
- [ ] Farmer alert dashboard *(stretch goal)*
- [ ] Full field trial and validation *(stretch goal)*

---

## Use Cases

- **Smallholder Farms** — Affordable early warning system replacing manual scouting walks
- **Large-Scale Agriculture** — Automated coverage of fields too large for frequent manual inspection
- **Research & Breeding Programs** — Systematic disease monitoring across experimental plots
- **Extension Services** — Tool for agricultural advisors to demonstrate disease identification

---

## Contributing

Contributions are welcome across all areas of the project — ML modeling, embedded systems, robotics firmware, and dashboard development. Please open an issue to discuss proposed changes before submitting a pull request.

---

*Built at the intersection of precision agriculture, computer vision, and autonomous systems.*
