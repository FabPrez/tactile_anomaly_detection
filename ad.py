import os
import cv2
import numpy as np
import open3d as o3d
from datetime import datetime
import sys
# === Percorsi ===
anomaly_detection_dir = os.path.dirname(os.path.abspath(__file__))
gsrobotics_dir = os.path.join(anomaly_detection_dir, '..', 'gsrobotics')
nn_model_path = os.path.join(gsrobotics_dir, 'models', 'nnmini.pt')
save_dir = gsrobotics_dir
sys.path.append(gsrobotics_dir)
sys.path.append(anomaly_detection_dir)


from utilities.gelsightmini import GelSightMini
from utilities.reconstruction import Reconstruction3D
from utilities.visualization import Visualize3D



# === Salvataggio point cloud ===
def save_pointcloud_from_depth(depth_map, output_dir, filename_prefix="pointcloud_gs_mini"):
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
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.ply"
    save_path = os.path.join(output_dir, filename)

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"✅ Point cloud salvata in: {save_path}")
    return save_path

# === Salvataggio immagine RGB ===
def save_rgb_image(image_rgb, output_dir, filename_prefix="image_gs_mini"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)
    print(f"✅ Immagine RGB salvata in: {save_path}")
    return save_path

# === Caricamento e visualizzazione point cloud ===
def load_and_visualize_pointcloud(pcd_path: str):
    if not os.path.isfile(pcd_path):
        print(f"❌ Il file non esiste: {pcd_path}")
        return

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            print(f"⚠️ La point cloud è vuota: {pcd_path}")
            return

        print(f"✅ Point cloud caricata con {len(pcd.points)} punti.")
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")
    except Exception as e:
        print(f"❌ Errore nel caricamento della point cloud: {e}")

# === Main live viewer ===
def main():
    camera_width, camera_height = 320, 240
    marker_mask_min, marker_mask_max = 0, 70
    use_gpu = False

    cam = GelSightMini(target_width=camera_width, target_height=camera_height)
    cam.select_device(device_idx=1)  # Cambia a 0 se necessario
    cam.start()

    reconstruction = Reconstruction3D(
        image_width=camera_width,
        image_height=camera_height,
        use_gpu=use_gpu
    )

    if reconstruction.load_nn(nn_model_path) is None:
        print("❌ Errore: modello non caricato")
        return

    visualizer3D = Visualize3D(
        pointcloud_size_x=camera_width,
        pointcloud_size_y=camera_height,
        save_path="",
        window_width=int(3 * camera_width),
        window_height=int(3 * camera_height)
    )

    print("Premi 'q' per uscire, 's' per salvare immagine + point cloud.")

    while True:
        frame = cam.update(dt=0)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth_map, _, grad_x, grad_y = reconstruction.get_depthmap(
            image=frame,
            markers_threshold=(marker_mask_min, marker_mask_max)
        )

        # visualizer3D.update(depth_map, gradient_x=grad_x, gradient_y=grad_y)
        # cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_rgb_image(frame, save_dir)
            save_pointcloud_from_depth(depth_map, save_dir)

    cam.camera.release()
    cv2.destroyAllWindows()
    visualizer3D.visualizer.destroy_window()

# === Avvio ===
if __name__ == "__main__":
    main()
