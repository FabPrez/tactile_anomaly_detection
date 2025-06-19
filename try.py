from gscaputre import GelSightCapture
import os
import time
import cv2

# === Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
gsrobotics_dir = os.path.join(script_dir, '..', 'gsrobotics')
model_path = os.path.join(gsrobotics_dir, 'models', 'nnmini.pt')
save_dir = gsrobotics_dir

gs = GelSightCapture(model_path=model_path)

try:
    print("⏳ Aspetto 10 secondi...")
    time.sleep(10)

    if gs.latest_frame is not None and gs.latest_depth is not None:
        gs.save_frame(gs.latest_frame, gs.latest_depth,save_dir)
    else:
        print("⚠️ Nessun frame valido disponibile.")

finally:
    gs.release()
