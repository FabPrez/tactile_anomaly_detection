import os
import cv2
import numpy as np
import open3d as o3d
from datetime import datetime
import threading
import sys
# === Percorsi ===
anomaly_detection_dir = os.path.dirname(os.path.abspath(__file__))
gsrobotics_dir = os.path.join(anomaly_detection_dir, '..','..', 'gsrobotics')
nn_model_path = os.path.join(gsrobotics_dir, 'models', 'nnmini.pt')
# save_dir = gsrobotics_dir
sys.path.append(gsrobotics_dir)
sys.path.append(anomaly_detection_dir)

from utilities.gelsightmini import GelSightMini
from utilities.reconstruction import Reconstruction3D

class GelSightCapture:
    def __init__(
        self,
        model_path: str,
        camera_width: int = 320,
        camera_height: int = 240,
        marker_mask_min: int = 0,
        marker_mask_max: int = 70,
        use_gpu: bool = False,
        camera_index: int = 1
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.marker_mask_min = marker_mask_min
        self.marker_mask_max = marker_mask_max

        self.cam = GelSightMini(target_width=camera_width, target_height=camera_height)
        self.cam.select_device(device_idx=camera_index)
        self.cam.start()

        self.reconstruction = Reconstruction3D(
            image_width=camera_width,
            image_height=camera_height,
            use_gpu=use_gpu
        )

        if self.reconstruction.load_nn(model_path) is None:
            raise RuntimeError("❌ Errore: modello non caricato")

        self.latest_frame = None
        self.latest_depth = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update_loop)
        self._thread.daemon = True
        self._thread.start()

        print("✅ GelSightCapture inizializzato con thread di acquisizione.")

    def _update_loop(self):
        while not self._stop_event.is_set():
            frame = self.cam.update(dt=0) # is already in rgb!!!!
            if frame is None:
                continue

            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            depth_map, _, _, _ = self.reconstruction.get_depthmap(
                image=frame,
                markers_threshold=(self.marker_mask_min, self.marker_mask_max)
            )

            self.latest_frame = frame
            self.latest_depth = depth_map

    def save_frame(self, frame_rgb, depth_map, save_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salva immagine RGB
        image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(save_dir,'rgb', f"image_gs_mini_{timestamp}.png")
        cv2.imwrite(img_path, image_bgr)
        print(f"✅ Immagine RGB salvata: {img_path}")

        # Salva point cloud
        h, w = depth_map.shape
        fx, fy = w / 2, h / 2
        cx, cy = w / 2, h / 2

        points = []
        for y in range(h):
            for x in range(w):
                z = depth_map[y, x]
                if not np.isfinite(z):
                    continue
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                points.append([X, Y, z])

        if not points:
            print("⚠️ Point cloud vuota, non salvata.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pc_path = os.path.join(save_dir, 'pointcloud', f"pointcloud_gs_mini_{timestamp}.ply")
        o3d.io.write_point_cloud(pc_path, pcd)
        print(f"✅ Point cloud salvata: {pc_path}")

    def release(self):
        self._stop_event.set()
        self._thread.join()
        if self.cam.camera:
            self.cam.camera.release()
        cv2.destroyAllWindows()
        print("🛑 Risorse rilasciate e thread fermato.")

