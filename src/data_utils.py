import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

# Obtém a raiz do projeto (uma pasta acima da pasta 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caminho exato confirmado pelo teste
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "FORTH_TRACE_DATASET-master")


def load_participant(participant_id, data_dir=DATA_DIR):
    pattern = os.path.join(data_dir, f"part{participant_id}", f"part{participant_id}dev*.csv")
    files = sorted(glob(pattern))

    if not files:
        print(f"Nenhum ficheiro encontrado para o participante {participant_id}. Caminho testado: {pattern}")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        df.columns = [
            "device", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "timestamp", "activity"
        ]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["participant"] = participant_id
    return df


def compute_magnitudes(df):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
    df["mag_mag"] = np.sqrt(df["mag_x"]**2 + df["mag_y"]**2 + df["mag_z"]**2)
    return df