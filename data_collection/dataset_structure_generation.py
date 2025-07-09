import os

# === Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir,'..','dataset')
BASE_DATASET_DIR = dataset_dir #! change the directory of the dataset folder here!

#! LEGEND
#PZ1 -> CODICE: -> DESCRIZIONE
#PZ2 -> CODICE: -> DESCRIZIONE
#PZ3 -> CODICE: -> DESCRIZIONE

def create_structure_for_PZ1(base_dir=BASE_DATASET_DIR):
    pezzo_name = "PZ1"
    posizioni = ["pos1", "pos2", "pos3", "pos4"]
    subfolders = ["rgb", "pointcloud"]

    pezzo_path = os.path.join(base_dir, pezzo_name)
    os.makedirs(pezzo_path, exist_ok=True)

    for posizione in posizioni:
        posizione_path = os.path.join(pezzo_path, posizione)
        for subfolder in subfolders:
            folder_path = os.path.join(posizione_path, subfolder)
            os.makedirs(folder_path, exist_ok=True)

    return pezzo_path

def create_structure_for_PZ2(base_dir=BASE_DATASET_DIR):
    pezzo_name = "PZ2"
    posizioni = [f"pos{i}" for i in range(1, 10)]  # pos1 to pos 9
    subfolders = ["rgb", "pointcloud"]
    pezzo_path = os.path.join(base_dir, pezzo_name)
    os.makedirs(pezzo_path, exist_ok=True)

    for posizione in posizioni:
        posizione_path = os.path.join(pezzo_path, posizione)
        for subfolder in subfolders:
            folder_path = os.path.join(posizione_path, subfolder)
            os.makedirs(folder_path, exist_ok=True)

    return pezzo_path

def get_dataset_path(pezzo, base_dir=BASE_DATASET_DIR):
    if pezzo == "PZ1":
        return create_structure_for_PZ1(base_dir)
    elif pezzo == "PZ2":
        return create_structure_for_PZ2(base_dir)
    else:
        raise ValueError(f"Pezzo '{pezzo}' non riconosciuto.")
    
def get_dataset_root_path(pezzo, base_dir=BASE_DATASET_DIR):
    """
    Restituisce il percorso assoluto alla cartella root del dataset per un dato pezzo,
    senza creare alcuna directory.
    """
    pezzo_name = pezzo.upper()
    if pezzo_name.startswith("PZ"):
        return os.path.join(base_dir, pezzo_name)
    else:
        raise ValueError(f"Formato pezzo non valido: '{pezzo}' (usa 'PZ1', 'PZ2', ecc.)")


# Esempio d'uso
if __name__ == "__main__":
    path = get_dataset_path("PZ2") # PZ1 | PZ2 | PZ3 | PZ4
    print(f"Struttura creata per 'PZ2' in: {os.path.abspath(path)}")
    
