# Tactile Anomaly Detection with GelSight

Tools and experiments for anomaly detection using tactile data from **GelSight** sensors.  
Target environment: **Windows**.

---

## Workspace Setup
**Create your workspace folder (e.g., `AD`)**:
```bash
mkdir AD
cd AD
```
Clone this repo:
```bash
git clone https://github.com/FabPrez/tactile_anomaly_detection.git
```

## Option A — Data Collection

Clone **gsrobotics**, create a virtual environment **`dc_venv`**, and install requirements:

Enter in the workspace and clone the repository for ther Gelsight mini:
```bash
cd AD
git clone https://github.com/gelsightinc/gsrobotics.git
```
Create a virtual environemnt and activate it:
```bash
python -m venv dc_venv
.\dc_venv\Scripts\activate
```
install the requirements:
```bash
cd gsrobotics
pip install -r requirements.txt
```

Use gsrobotics to connect to the sensor and record data.


## Option B - Training & Testing
Enter in the workspace, create the virtual environment and activate it:
```bash
cd AD
python -m venv ad_venv
.\ad_venv\Scripts\activate
```

install the requirements:
```bash
cd tactile_anomaly_detection
pip install -r requirements.txt
```
