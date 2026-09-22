import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_participant_column(df):
    if 'participant' in df.columns:
        return 'participant'
    elif 'participant_id' in df.columns:
        return 'participant_id'
    else:
        raise KeyError("Coluna de participante não encontrada (esperado 'participant' ou 'participant_id').")

def split_within_subject(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    print(f"--- Divisão Intra-Sujeito (Stratified Shuffle) ---")
    
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("As proporções de treino, validação e teste devem somar 1.0")

    # 1. Separar Treino (60%) do Resto (40%)
    # Stratify garante que a proporção de atividades se mantém igual em todos os sets
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed, 
        stratify=df['activity']
    )

    # 2. Separar Validação (20% total) e Teste (20% total) a partir do Resto
    # Como o Resto é 40% do total, dividimos ao meio (0.5) para ter 20% e 20%
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=seed, 
        stratify=temp_df['activity']
    )

    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

def split_between_subjects(df, n_train=9, n_val=3, n_test=3, seed=42):
    print(f"--- Divisão Inter-Sujeitos (Split por Participante) ---")
    
    part_col = get_participant_column(df)
    unique_participants = df[part_col].unique()
    
    total_needed = n_train + n_val + n_test
    if len(unique_participants) < total_needed:
        print(f"Número de participantes disponíveis ({len(unique_participants)}) é menor que o solicitado ({total_needed}). Ajustando lógica...")
        np.random.seed(seed)
        shuffled = np.random.permutation(unique_participants)
        n = len(shuffled)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
    else:
        np.random.seed(seed)
        shuffled = np.random.permutation(unique_participants)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    print(f"IDs Treino ({len(train_ids)}): {train_ids}")
    print(f"IDs Validação ({len(val_ids)}): {val_ids}")
    print(f"IDs Teste ({len(test_ids)}): {test_ids}")

    train_df = df[df[part_col].isin(train_ids)]
    val_df = df[df[part_col].isin(val_ids)]
    test_df = df[df[part_col].isin(test_ids)]

    print(f"Amostras -> Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }