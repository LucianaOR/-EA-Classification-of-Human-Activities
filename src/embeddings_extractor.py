import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- 1. Carregamento do Modelo ---
def load_model():
    ''' Carrega o feature extractor do modelo HARNet (SSL). '''
    repo = 'OxWearables/ssl-wearables'
    model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
    
    feature_encoder = model.feature_extractor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_encoder.to(device)
    feature_encoder.eval()
    
    return feature_encoder, device

# --- 2. Reamostragem (Resample) ---
def resample_segment(acc_data, input_fs, target_fs=30.0, target_duration=5.0):
    target_samples = int(target_fs * target_duration)
    n_input = acc_data.shape[0]
    duration = n_input / input_fs
    t_in = np.linspace(0, duration, n_input)
    t_out = np.linspace(0, target_duration, target_samples)
    
    resampled = np.zeros((target_samples, 3), dtype=np.float32)
    for i in range(3):
        f = interp1d(t_in, acc_data[:, i], kind='linear', fill_value="extrapolate")
        resampled[:, i] = f(t_out)
        
    return resampled

# --- 3. Pipeline Principal (CORRIGIDO) ---
def generate_embeddings_dataset(df, input_fs=50.0, window_sec=5.0, stride_sec=2.5):
    print(f"--- Iniciando geração de Embeddings (Input FS: {input_fs}Hz -> Target: 30Hz) ---")
    
    feature_encoder, device = load_model()
    
    window_samples = int(input_fs * window_sec)
    stride_samples = int(input_fs * stride_sec)
    
    segments_buffer = []
    metadata_buffer = []
    
    group_cols = ['participant', 'device'] if 'device' in df.columns else ['participant_id']
    if isinstance(group_cols, str): group_cols = [group_cols]

    for ids, group in df.groupby(group_cols):
        group = group.sort_index()
        acc_raw = group[['acc_x', 'acc_y', 'acc_z']].values
        activities = group['activity'].values
        
        for i in range(0, len(group) - window_samples, stride_samples):
            seg_acc = acc_raw[i : i + window_samples]
            seg_act = activities[i : i + window_samples]
            seg_resampled = resample_segment(seg_acc, input_fs)
            
            try:
                vals, counts = np.unique(seg_act, return_counts=True)
                act_label = vals[np.argmax(counts)]
            except:
                continue 
                
            segments_buffer.append(seg_resampled)
            
            current_meta = {}
            if len(group_cols) > 1 and isinstance(ids, tuple):
                for k, col in enumerate(group_cols):
                    current_meta[col] = ids[k]
            else:
                current_meta[group_cols[0]] = ids
            current_meta['activity'] = act_label
            metadata_buffer.append(current_meta)

    if not segments_buffer:
        print("Nenhum segmento válido encontrado.")
        return None

    X_array = np.array(segments_buffer) 
    X_array = np.transpose(X_array, (0, 2, 1)) 
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)
    
    print(f"Extraindo embeddings para {len(X_tensor)} segmentos...")
    
    embeddings_list = []
    batch_size = 64 
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            emb = feature_encoder(batch)
            embeddings_list.append(emb.cpu().numpy())
            
    full_embeddings = np.concatenate(embeddings_list, axis=0)
    
    if full_embeddings.ndim > 2:
        full_embeddings = full_embeddings.squeeze()
        print(f"Shape ajustado para 2D: {full_embeddings.shape}")
    # ---------------------

    feat_cols = [f"emb_{k}" for k in range(full_embeddings.shape[1])]
    df_emb = pd.DataFrame(full_embeddings, columns=feat_cols)
    
    df_meta = pd.DataFrame(metadata_buffer)
    final_df = pd.concat([df_meta, df_emb], axis=1)
    
    print(f"Dataset de Embeddings gerado com shape: {final_df.shape}")
    return final_df