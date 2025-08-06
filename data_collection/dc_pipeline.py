import os
import time
import sys

# === Percorsi ===
data_collection_dir = os.path.dirname(os.path.abspath(__file__))
sensor_scripts_dir = os.path.join(data_collection_dir, '..')
# robot_interface_dir = os.path.join(data_collection_dir, 'robot_interface')

#print(robot_interface_dir)

# save_dir = gsrobotics_dir
sys.path.append(sensor_scripts_dir)
#sys.path.append(robot_interface_dir)



from robot_interface import robot_socket_conection
from sensor_scripts import gscapture
from dataset_structure_generation import get_dataset_root_path

def start_pipeline(pezzo_num, number_of_diff_poses, number_of_acquisition_per_pose,fault_data_acquisiton=False):
    sensor = gscapture.GelSightCapture()
    rsc = robot_socket_conection.RobotSocketInterface()
    rsc.select_pezzo(pezzo_num)  # Seleziona il pezzo
    pezzo_id = f'PZ{pezzo_num}'
    print(f"\n--- {pezzo_id} data collection started ---")
    pezzo_root_path = get_dataset_root_path(pezzo_id)
    
    # 1. Attendi "can start"
    print(">> Waiting can_meas...")
    if not rsc.wait_for_message("can_start"):
        print("Connessione interrotta durante attesa 'can_start'")
        return
    
    for pose_idx in range(1, number_of_diff_poses + 1):
        pose_name = f"pos{pose_idx}"
        save_path = os.path.join(pezzo_root_path, pose_name)
        
        if fault_data_acquisiton:
            save_path = os.path.join(pezzo_root_path,pose_name,'fault')
            
        print(f"\n--- Inizio acquisizione per {pose_name} ---")

        for acq_idx in range(1, number_of_acquisition_per_pose + 1):
            print(f"\n[{pose_name}] Acquisizione {acq_idx}/{number_of_acquisition_per_pose}")

            # 1. Attendi "can_meas"
            print(">> Waiting can_meas...")
            if not rsc.wait_for_message("can_meas"):
                print("Connessione interrotta durante attesa 'can_meas'")
                return

            # 2. Acquisizione
            print(">> Acquisizione immagine...")
            sensor.save_frame(sensor.latest_frame, sensor.latest_depth, save_path)
            print(">> Immagine acquisita...")
            # 3. Invia comando '9' per avanzare
            rsc.send_message("9")
            time.sleep(3)
            
            #TODO: bisognerebbe far allontanare il robot e poi far "ri-toccare" l'oggetto per ogni nuova misura
    # 4. Invia comando '8' per chiudere connessione
    if not rsc.wait_for_message("finish"):
                print("Connessione interrotta durante attesa 'finish'")
                return
    time.sleep(5)
    rsc.send_message("8")
    time.sleep(2)
    rsc.close()

    print(f"\n✅ Acquisizione completata per {pezzo_id}")
