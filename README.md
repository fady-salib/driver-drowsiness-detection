# Real-Time Driver Drowsiness Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%2520Vision-red.svg)](https://opencv.org/)
[![YOLO / Deep Learning](https://img.shields.io/badge/Model-YOLO%20%2F%20CNN-orange.svg)]()

An edge-optimized computer vision system designed to monitor driver alertness in real time, detect signs of fatigue or distraction (such as eye closure and yawning), and trigger immediate audio-visual warnings to prevent accidents.

---

## 📄 Table of Contents
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Detection Logic & Metrics](#-detection-logic--metrics)
* [Hardware Setup & Camera Integration](#-hardware-setup--camera-integration)
* [Quick-Start Guide](#-quick-start-guide)
* [Repository Structure](#-repository-structure)
* [License](#-license)

---

## 🛠 Key Features

* **Real-Time Video Ingestion**: Efficient frame processing from USB webcams or edge hardware (e.g., Raspberry Pi camera modules).
* **Facial Landmark & State Tracking**: Monitors key facial features to calculate Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
* **Deep Learning Object Detection**: Integrates lightweight models (like YOLO or customized CNN architectures) for robust face and feature localization under varying lighting conditions.
* **Instant Alert System**: Triggers buzzer modules, relay triggers, or software warnings when fatigue thresholds are breached over consecutive frames.

---

## 🏗 System Architecture

The pipeline captures live video frames, extracts facial regions, analyzes features, and triggers alerts if a drowsy state persists:
