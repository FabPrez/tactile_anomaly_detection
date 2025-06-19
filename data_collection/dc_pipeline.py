import os
import time
from tactile_anomaly_detection.robot_interface import robot_socket_conection
from sensor_scripts import gscapture
from dataset_structure_generation import get_dataset_root_path

def start_pipeline(pezzo_num, number_of_diff_poses, number_of_acquisition_per_pose):
    sensor = gscapture.GelSightCapture()
    rsc = robot_socket_conection.RobotSocketInterface()
    rsc.select_pezzo(pezzo_num)  # Seleziona il pezzo
    pezzo_id = f'PZ{pezzo_num}'
    print(f"\n--- {pezzo_id} data collection started ---")
    pezzo_root_path = get_dataset_root_path(pezzo_id)

    for pose_idx in range(1, number_of_diff_poses + 1):
        pose_name = f"pos{pose_idx}"
        save_path = os.path.join(pezzo_root_path, pose_name)
        print(f"\n--- Inizio acquisizione per {pose_name} ---")

        for acq_idx in range(1, number_of_acquisition_per_pose + 1):
            print(f"\n[{pose_name}] Acquisizione {acq_idx}/{number_of_acquisition_per_pose}")

            # 1. Attendi "can_meas"
            if not rsc.wait_for_message("can_meas"):
                print("Connessione interrotta durante attesa 'can_meas'")
                return

            # 2. Acquisizione
            print(">> Acquisizione immagine...")
            sensor.save_frame(save_path)

            # 3. Invia comando '9' per avanzare
            rsc.send_message("9")
    
    rsc.close()

    print(f"\n✅ Acquisizione completata per {pezzo_id}")
