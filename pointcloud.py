import os
import time
import cv2
import numpy as np
import open3d as o3d
import sys

anomaly_detection_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(anomaly_detection_dir)

gsrobotics_dir = os.path.join(anomaly_detection_dir,'..','gsrobotics') 
sys.path.append(gsrobotics_dir)
nn_model_path = os.path.join(gsrobotics_dir,'models','nnmini.pt')
import os
import sys
import numpy as np
import cv2
import open3d as o3d

from utilities.reconstruction import Reconstruction3D
from utilities.gelsightmini import GelSightMini


def generate_point_cloud(depth_map: np.ndarray, scale: float = 1.0) -> o3d.geometry.PointCloud:
    h, w = depth_map.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    x = xx * scale
    y = yy * scale
    z = depth_map

    valid = ~np.isnan(z)
    x, y, z = x[valid], y[valid], z[valid]

    points = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def main():
    # Setup paths
    anomaly_detection_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(anomaly_detection_dir)

    gsrobotics_dir = os.path.join(anomaly_detection_dir, '..', 'gsrobotics')
    sys.path.append(gsrobotics_dir)

    nn_model_path = os.path.join(gsrobotics_dir, 'models', 'nnmini.pt')

    # Init camera
    cam = GelSightMini(target_width=320, target_height=240)
    cam.select_device(1)
    cam.start()

    # Init depth estimator
    recon = Reconstruction3D(image_width=320, image_height=240, use_gpu=False)
    if recon.load_nn(nn_model_path) is None:
        print("Model loading failed.")
        return

    # Init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Real-Time Point Cloud")

    # Dummy geometry for update
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    try:
        while True:
            frame = cam.update(dt=0)
            if frame is None:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            depth_map, _, _, _ = recon.get_depthmap(image=image, markers_threshold=(20, 255))
            if depth_map is None or np.isnan(depth_map).all():
                continue

            # Update point cloud
            new_pcd = generate_point_cloud(depth_map)

            pcd.points = new_pcd.points
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        cam.camera.release()
        vis.destroy_window()


if __name__ == "__main__":
    main()
