#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import torch

from feature_extraction import extract_multivariate_features
from embeddings_extractor import load_model, resample_segment

print("="*80)
print("TESTE DE DIAGNÓSTICO DO MODELO")
print("="*80)

print("\n[1] Carregar modelo...")
with open('best_model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

config = artifacts['config']
model = artifacts['model']

print(f"  Dataset: {config['dataset_type']}")
print(f"  Scenario: {config['scenario']}")
print(f"  Best k: {config['best_k']}")
print(f"  Classes do modelo: {model.classes_}")
print(f"  Número de features: {len(config['final_feature_names'])}")

print("\n[2] Carregar dados de teste...")
df = pd.read_csv('/Users/pedro/Documents/Faculdade/ECAC/Projeto ECAC/FORTH_TRACE_DATASET-master - cópia/part0/part0dev1.csv')
print(f"  Shape: {df.shape}")
print(f"  Atividades únicas: {sorted(df['activity'].unique())}")

print("\n[3] Testar predições...")

cols_sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
activity_map = {
    1: "Stand", 2: "Sit", 3: "Sit and Talk", 4: "Walk", 
    5: "Walk and Talk", 6: "Climb Stair", 7: "Climb Stair and Talk"
}

all_predictions = []
all_true_labels = []

if config['dataset_type'] == 'embeddings':
    print("  Carregar modelo de embeddings...")
    feature_encoder, device = load_model()

for activity_id in [1, 4, 6]:  
    samples = df[df['activity'] == activity_id]
    if len(samples) < 256:
        print(f"  ⚠ Atividade {activity_id}: dados insuficientes")
        continue
    
    for i in range(3):
        start_idx = i * 256
        window = samples.iloc[start_idx:start_idx+256][cols_sensors].values
        
        if config['dataset_type'] == 'embeddings':
            acc_data = window[:, :3]
            
            INPUT_FS = 50.0
            acc_resampled = resample_segment(acc_data, INPUT_FS, target_fs=30.0, target_duration=5.0)
            acc_transposed = acc_resampled.T
            
            X_tensor = torch.tensor(acc_transposed, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = feature_encoder(X_tensor)
            
            emb_array = emb.cpu().numpy().squeeze()
            if emb_array.ndim > 1:
                emb_array = emb_array.flatten()
            
            X_processed = pd.DataFrame([emb_array])
        else:
            df_window = pd.DataFrame(window, columns=cols_sensors)
            features_dict = extract_multivariate_features(df_window)
            X_processed = pd.DataFrame([features_dict])
            X_processed = X_processed.reindex(columns=config['feature_names'], fill_value=0)
        
        if 'scaler' in artifacts:
            X_scaled = artifacts['scaler'].transform(X_processed)
            X_processed = pd.DataFrame(X_scaled)
        
        if 'pca' in artifacts:
            X_pca = artifacts['pca'].transform(X_processed)
            X_processed = pd.DataFrame(X_pca)
        elif 'selector' in artifacts:
            X_selected = artifacts['selector'].transform(X_processed)
            X_processed = pd.DataFrame(X_selected)
        
        pred = model.predict(X_processed)[0]
        all_predictions.append(pred)
        all_true_labels.append(activity_id)

print("\n[4] Resultados:")
pred_counts = Counter(all_predictions)

print(f"\nTotal de testes: {len(all_predictions)}")
print("Distribuição das predições:")
for act_id, count in sorted(pred_counts.items()):
    pct = count / len(all_predictions) * 100
    print(f"  Atividade {act_id} ({activity_map.get(act_id, 'Unknown')}): {count} ({pct:.1f}%)")

print("\nMatriz de confusão (real vs previsto):")
for i, true_label in enumerate(all_true_labels):
    pred_label = all_predictions[i]
    match = "✓" if true_label == pred_label else "✗"
    print(f"  {match} Real: {true_label} ({activity_map[true_label]}) -> Previsto: {pred_label} ({activity_map.get(pred_label, 'Unknown')})")

print("\n" + "="*80)
accuracy = sum(1 for i in range(len(all_true_labels)) if all_true_labels[i] == all_predictions[i]) / len(all_true_labels)
print(f"Acurácia: {accuracy:.2%}")

if accuracy < 0.5:
    print("\n PROBLEMA DETETADO!")
    most_common_pred = pred_counts.most_common(1)[0][0]
    if pred_counts[most_common_pred] > len(all_predictions) * 0.7:
        print(f"   Modelo sempre prevê: {activity_map.get(most_common_pred, 'Unknown')}")
        print("   Possível causa: Features colapsando para valores similares")
else:
    print("\n Modelo funcionando corretamente!")

print("="*80)
