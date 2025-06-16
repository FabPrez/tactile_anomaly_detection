# Tactile Anomaly Detection with GelSight

This repository contains tools and experiments for performing anomaly detection using tactile data acquired from GelSight sensors.


##  Installation
 **Note:** This setup guide is designed for **Windows environments**.

Follow these steps to set up your workspace:

**Create your workspace folder (e.g., `AD`)**:
 ```bash
 mkdir AD
 cd AD
 ```
Clone the main anomaly detection repository:

```bash
git clone https://github.com/FabPrez/tactile_anomaly_detection.git
```
Clone the GelSight robotics library:

```bash
git clone https://github.com/gelsightinc/gsrobotics.git
```
Create and activate a virtual environment:
```bash
python -m venv ad_venv
.\ad_venv\Scripts\activate
```
Install the requirements from gsrobotics:

```bash
cd gsrobotics
pip install -r requirements.txt
```
