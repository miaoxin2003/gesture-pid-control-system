# AGENTS.md

## Project Overview

Dual-mode gesture recognition system: **Python** (MediaPipe + SVM) runs on PC, sends coordinates via serial → **STM32F103ZE** firmware runs PID control loop.

## Directory Structure

- `control/` - STM32F103ZE Keil MDK firmware project
  - `USER/main.c` - Firmware entry point
  - `HARDWARE/PID/` - PID control implementation
  - `HARDWARE/serial/` - Serial receive protocol (`#X$Y\r\n`)
  - `HARDWARE/TIMER/` - TIM3 PWM output (channels 1, 2)
  - `HARDWARE/LED/`, `HARDWARE/KEY/` - Peripherals
  - `STM32F10x_FWLib/` - STM32 standard peripheral library
  - `SYSTEM/` - delay, sys, usart utilities
- `svm/` - Python gesture recognition
  - `inference.py` - Real-time gesture detection + serial output
  - `train_model.py` - SVM training script
  - `collect_data.py` - Data collection for training
  - `dataset.csv` - Training dataset
  - `HandTrackingModule.py` - 手部检测+**自适应EMA滤波**（速度自适应alpha + 离群点检测）+ **预测控制**（线性速度外推）+ **手势状态机**（保持确认防误触）

## Build & Run

### Firmware (control/)
- Open `control/USER/CONTROL.uvprojx` in **Keil MDK uVision**
- Build targets: USART (STM32F103ZE, Cortex-M3)
- Output: `control/OBJ/CONTROL.hex` or `.axf`
- Flash to device via ST-Link/J-Link debugger

### Python (svm/)
```bash
pip install opencv-python mediapipe scikit-learn joblib numpy
python inference.py        # Real-time gesture recognition
python train_model.py      # Train SVM classifier
python collect_data.py     # Collect training data
```

## Serial Protocol

- **Baud rate**: 115200
- **Format**: `#X坐标$Y坐标\r\n`
  - `#` = start delimiter
  - `$` = X/Y separator
  - Firmware parses digits only, ignores other characters
- **Firmware receives**: via USART_RX_BUF, sets `coords[0]` (X), `coords[1]` (Y)

## PID Control (main loop)

1. Receive coordinates via `recieveData()`
2. Run PID: `pwmval_x = pwmval_x + pid(coords[0], targetX, &PID_x)`
3. Output PWM: `TIM_SetCompare1(TIM3, pwmval_x)`, `TIM_SetCompare2(TIM3, pwmval_y)`

- TIM3 PWM freq: 50Hz (ARR=9999, PSC=143)
- PWM range: X=[300,1200], Y=[300,1000]

## Key Constants

- Target coordinates: `targetX=320`, `targetY=240`
- PID gains (PID_x): `Kp=0.04, Ki=0, Kd=0.30`
- PID gains (PID_y): `Kp=0.05, Ki=0, Kd=0.30`
