import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_extraction import reliefF_optimized 

def process_split_scenarios(split_dict, dataset_name="Dataset"):
    """
    Recebe um dicionário com chaves 'train', 'val', 'test' (DataFrames).
    Retorna um dicionário com os 3 cenários (A, B, C) processados.
    
    Garante que fit() é feito APENAS no treino.
    """
    print(f"\n A processar cenários para: {dataset_name}...")
    
    ignore_cols = ['participant', 'participant_id', 'activity', 'device']
    
    def separate_xy(df):
        meta_cols = [c for c in df.columns if c in ignore_cols]
        y = df['activity']
        X = df.drop(columns=meta_cols).select_dtypes(include=[np.number])
        return X, y, meta_cols

    X_train, y_train, meta_cols = separate_xy(split_dict['train'])
    X_val, y_val, _ = separate_xy(split_dict['val'])
    X_test, y_test, _ = separate_xy(split_dict['test'])

    meta_train = split_dict['train'][meta_cols]
    meta_val = split_dict['val'][meta_cols]
    meta_test = split_dict['test'][meta_cols]


    print("   1. Normalização (StandardScaler)...")
    scaler = StandardScaler()
    scaler.fit(X_train) 
    
    X_train_norm = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_norm = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # ==========================================================
    # CENÁRIO A: All Features (Apenas Normalizado)
    # ==========================================================
    scenario_A = {
        'train': (X_train_norm, y_train),
        'val': (X_val_norm, y_val),
        'test': (X_test_norm, y_test)
    }

    # ==========================================================
    # CENÁRIO B: PCA (90% Variância)
    # ==========================================================
    print("   2. PCA (90% Variância)...")
    pca = PCA(n_components=0.90) 
    pca.fit(X_train_norm) 
    
    cols_pca = [f"PC{i+1}" for i in range(pca.n_components_)]
    
    X_train_pca = pd.DataFrame(pca.transform(X_train_norm), columns=cols_pca, index=X_train_norm.index)
    X_val_pca = pd.DataFrame(pca.transform(X_val_norm), columns=cols_pca, index=X_val_norm.index)
    X_test_pca = pd.DataFrame(pca.transform(X_test_norm), columns=cols_pca, index=X_test_norm.index)

    scenario_B = {
        'train': (X_train_pca, y_train),
        'val': (X_val_pca, y_val),
        'test': (X_test_pca, y_test),
        'explained_variance': pca.explained_variance_ratio_
    }
    print(f"Reduzido para {len(cols_pca)} componentes.")

    # ==========================================================
    # CENÁRIO C: Feature Selection (ReliefF - Top 15)
    # ==========================================================
    print("   3. Feature Selection (ReliefF - Top 15)...")
    try:
        scores = reliefF_optimized(X_train_norm, y_train, n_neighbors=10)
        top_15_features = scores.head(15).index.tolist()
        
        X_train_relief = X_train_norm[top_15_features]
        X_val_relief = X_val_norm[top_15_features]
        X_test_relief = X_test_norm[top_15_features]
        
        scenario_C = {
            'train': (X_train_relief, y_train),
            'val': (X_val_relief, y_val),
            'test': (X_test_relief, y_test),
            'selected_features': top_15_features
        }
        print(f"Features selecionadas: {top_15_features}")
        
    except Exception as e:
        print(f"Erro no ReliefF: {e}")
        scenario_C = None

    return {
        'A_all_features': scenario_A,
        'B_pca': scenario_B,
        'C_relief': scenario_C
    }