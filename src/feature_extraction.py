import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def _calculate_features(window, sampling_rate=50):
    features = {
        "mean": np.mean(window),
        "std": np.std(window),
        "variance": np.var(window),
        "min": np.min(window),
        "max": np.max(window),
        "median": np.median(window),
        "amplitude": np.max(window) - np.min(window),
        "kurtosis": kurtosis(window),
        "skewness": skew(window),
        "rms": np.sqrt(np.mean(np.square(window))),
        "zcr": np.sum(np.diff(np.sign(window)) != 0) / len(window)
    }

    fft_values = np.abs(fft(window))
    fft_power = fft_values**2
    freqs = np.fft.fftfreq(len(window), 1 / sampling_rate)

    fft_power_sum = np.sum(fft_power) + 1e-12
    normalized_power = fft_power / fft_power_sum
    
    features.update({
        "energy": np.sum(fft_power),
        "dominant_frequency": freqs[np.argmax(fft_values)],
        "entropy": -np.sum(normalized_power * np.log2(normalized_power + 1e-12)),
        "bandwidth": np.sqrt(np.sum(((freqs - np.sum(freqs * normalized_power))**2) * normalized_power)),
        "centroid": np.sum(freqs * fft_values) / (np.sum(fft_values) + 1e-12)
    })

    return features

def extract_multivariate_features(window_df, sampling_rate=50):
    all_features = {}
    
    raw_axes = [
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z'
    ]

    # 1. Extrai features de cada um dos 9 eixos brutos
    for col in raw_axes:
        if col in window_df.columns:
            signal = window_df[col].values
            features = _calculate_features(signal, sampling_rate)
            for key, value in features.items():
                all_features[f"{col}_{key}"] = value

    # 2. Calcula magnitudes e extrai features delas
    df_temp = window_df.copy()
    if all(c in df_temp for c in ['acc_x', 'acc_y', 'acc_z']):
        df_temp["acc_mag"] = np.sqrt(df_temp["acc_x"]**2 + df_temp["acc_y"]**2 + df_temp["acc_z"]**2)
    if all(c in df_temp for c in ['gyro_x', 'gyro_y', 'gyro_z']):
        df_temp["gyro_mag"] = np.sqrt(df_temp["gyro_x"]**2 + df_temp["gyro_y"]**2 + df_temp["gyro_z"]**2)
    if all(c in df_temp for c in ['mag_x', 'mag_y', 'mag_z']):
        df_temp["mag_mag"] = np.sqrt(df_temp["mag_x"]**2 + df_temp["mag_y"]**2 + df_temp["mag_z"]**2)
    
    magnitude_cols = [c for c in ["acc_mag", "gyro_mag", "mag_mag"] if c in df_temp.columns]

    for col in magnitude_cols:
        signal = df_temp[col].values
        features = _calculate_features(signal, sampling_rate)
        for key, value in features.items():
            all_features[f"{col}_{key}"] = value

    return all_features

def extract_features_pipeline_multivariate(df, window_size=100, stride=50, sampling_rate=50):
    if df is None or df.empty:
        print("O DataFrame de entrada para extração de features está vazio ou é nulo.")
        return None

    final_feature_set = []

    for participant, participant_df in df.groupby('participant'):
        for device, device_df in participant_df.groupby('device'):

            device_df = device_df.reset_index(drop=True)

            windows_indices = range(0, len(device_df) - window_size + 1, stride)
            
            print(f"Processando Participante {participant}, Dispositivo {device}: {len(windows_indices)} janelas encontradas.")

            for i in windows_indices:
                window_df = device_df.iloc[i:i + window_size]
                
                if window_df.empty or len(window_df) < window_size:
                    continue
                try:
                    activity_label = window_df['activity'].mode()[0]
                except IndexError:
                    continue
                
                features = extract_multivariate_features(window_df, sampling_rate)
                
                features["participant"] = participant
                features["device"] = device
                features["activity"] = activity_label
                
                final_feature_set.append(features)

    if not final_feature_set:
        print("Nenhuma feature foi extraída de nenhum participante.")
        return None

    feature_set_df = pd.DataFrame(final_feature_set)
    
    cols_order = ['participant', 'device', 'activity'] + [c for c in feature_set_df.columns if c not in ['participant', 'device', 'activity']]
    
    return feature_set_df[cols_order]

def normalize_features(feature_set):
    scaler = StandardScaler()
    feature_columns = feature_set.select_dtypes(include=[np.number]).columns
    norm_features = scaler.fit_transform(feature_set[feature_columns])
    return pd.DataFrame(norm_features, columns=feature_columns, index=feature_set.index)

def apply_pca_with_analysis(feature_set, n_components=None, explained_variance_threshold=0.95):
    feature_columns = feature_set.select_dtypes(include=[np.number]).columns
    X = feature_set[feature_columns].values
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    selected_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1 if np.any(cumulative_variance >= explained_variance_threshold) else len(explained_variance)

    component_columns = [f"PC{i+1}" for i in range(principal_components.shape[1])]
    transformed_features = pd.DataFrame(principal_components, columns=component_columns, index=feature_set.index)

    return transformed_features, explained_variance, selected_components

def fisher_score(features, labels):
    scores = []
    feature_columns = features.columns
    overall_mean = features.mean()

    for feature in feature_columns:
        numerator = 0
        denominator = 0
        for label in np.unique(labels):
            class_samples = features[labels == label]
            n_c = len(class_samples)
            if n_c > 1: 
                mean_c = class_samples[feature].mean()
                var_c = class_samples[feature].var()
                numerator += n_c * (mean_c - overall_mean[feature]) ** 2
                denominator += n_c * var_c
        
        scores.append(numerator / denominator if denominator != 0 else 0)

    return pd.Series(scores, index=feature_columns).sort_values(ascending=False)

def reliefF_optimized(features, labels, n_neighbors=10):
    n_samples, n_features = features.shape
    feature_weights = np.zeros(n_features)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(features)
    indices = nn.kneighbors(features, n_neighbors=n_neighbors + 1)[1][:, 1:]

    labels_array = np.array(labels)
    same_class_mask = (labels_array[indices] == labels_array[:, None])
    
    for f in range(n_features):
        feature_col = features.iloc[:, f].values
        diffs = np.abs(feature_col[:, None] - feature_col[indices])
        
        same_class_sum = np.sum(same_class_mask, axis=1)
        diff_class_sum = n_neighbors - same_class_sum

        term_miss = np.sum(diffs * ~same_class_mask, axis=1, where=diff_class_sum[:, None] > 0) / (diff_class_sum + 1e-9)
        term_hit = np.sum(diffs * same_class_mask, axis=1, where=same_class_sum[:, None] > 0) / (same_class_sum + 1e-9)
        
        feature_weights[f] = np.sum(term_miss - term_hit)

    feature_weights /= n_samples
    return pd.Series(feature_weights, index=features.columns).sort_values(ascending=False)

def identify_top_features(features, labels, n_top=10):
    fisher_scores = fisher_score(features, labels)
    relief_scores = reliefF_optimized(features, labels)
    return fisher_scores.head(n_top).index.tolist(), relief_scores.head(n_top).index.tolist()

def compare_top_features(top_fisher, top_relief):
    set_fisher = set(top_fisher)
    set_relief = set(top_relief)
    common_features = list(set_fisher.intersection(set_relief))
    fisher_exclusive = list(set_fisher - set_relief)
    relief_exclusive = list(set_relief - set_fisher)
    return common_features, fisher_exclusive, relief_exclusive

def get_compressed_features(features, top_features):
    return features[top_features]

def explain_feature_selection():
    explanation = """
--- Vantagens e Limitações da Seleção de Features ---

Vantagens:
- Reduz a complexidade do modelo e o risco de overfitting.
- Melhora o desempenho computacional (treinamento e inferência mais rápidos).
- Mantém a interpretabilidade das features originais, ao contrário de métodos como PCA.

Limitações:
- Métodos como Fisher Score avaliam features individualmente, podendo perder interações valiosas.
- ReliefF é computacionalmente mais caro que métodos univariados.
- A escolha do 'melhor' subconjunto de features não é garantida e pode variar com os dados.
"""
    print(explanation)

def custom_smote(X, y, minority_class, num_synthetic_samples_to_generate, n_neighbors=5):
    # 1. Filtrar apenas a classe minoritária
    X_minority = X[y == minority_class].values
    
    n_minority_samples = len(X_minority)
    
    actual_k = min(n_neighbors, n_minority_samples - 1)
    
    if actual_k < 1:
        print(f"Impossível aplicar SMOTE. Classe {minority_class} tem apenas {n_minority_samples} amostra(s).")
        return X, y

    # 2. Encontrar os k vizinhos mais próximos
    nn = NearestNeighbors(n_neighbors=actual_k)
    nn.fit(X_minority)
    
    synthetic_samples_list = []
    synthetic_labels_list = []
    
    for _ in range(num_synthetic_samples_to_generate):
        idx_i = np.random.randint(0, n_minority_samples)
        x_i = X_minority[idx_i]
        
        distances, indices = nn.kneighbors(x_i.reshape(1, -1), n_neighbors=actual_k + 1)
        
        neighbor_indices = indices[0][1:]
        
        idx_j = np.random.choice(neighbor_indices)
        x_j = X_minority[idx_j]
        
        alpha = np.random.rand()
        x_new = x_i + alpha * (x_j - x_i)
        
        synthetic_samples_list.append(x_new)
        synthetic_labels_list.append(minority_class)
        
    # 3. Concatenar
    if len(synthetic_samples_list) > 0:
        X_synthetic = pd.DataFrame(synthetic_samples_list, columns=X.columns)
        y_synthetic = pd.Series(synthetic_labels_list, name=y.name)
        
        X_resampled = pd.concat([X, X_synthetic], ignore_index=True)
        y_resampled = pd.concat([y, y_synthetic], ignore_index=True)
        
        return X_resampled, y_resampled
    else:
        return X, y



