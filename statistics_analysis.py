import numpy as np
from scipy import stats
import pandas as pd 
# -------------------------------
# 4.1 - Análise de Significância Estatística
# -------------------------------
def analyze_statistical_significance(df, variable_name, label):
    print(f"\n--- 4.1 - Análise de Significância Estatística para {label} ({variable_name}) ---")

    activity_map_names = {
        1: "Stand", 2: "Sit", 3: "Sit and Talk", 4: "Walk", 5: "Walk and Talk",
        6: "Climb Stair (up/down)", 7: "Climb Stair (up/down) and talk",
        8: "Stand-> Sit", 9: "Sit-> Stand", 10: "Stand-> Sit and talk",
        11: "Sit->Stand and talk", 12: "Stand-> walk", 13: "Walk-> stand",
        14: "Stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk",
        15: "Climb stairs (up/down) -> walk",
        16: "Climb stairs (up/down) and talk -> walk and talk"
    }
    
    df_temp = df.copy()
    if 'activity' in df_temp.columns:
        df_temp['activity_name'] = df_temp['activity'].map(activity_map_names).fillna(df_temp['activity'])
        group_column = 'activity_name'
    else:
        print("Aviso: Coluna 'activity_label' não encontrada. Verifique o nome da coluna de atividades.")
        return

    # 1. Preparar os dados por atividade
    activity_data_tuples = [(name, group[variable_name].dropna().values) 
                            for name, group in df_temp.groupby(group_column)]
    
    activity_data_tuples = [(name, data) for name, data in activity_data_tuples if len(data) >= 2]

    if not activity_data_tuples:
        print("Nenhum dado válido ou atividades suficientes encontradas para análise de significância após filtragem.")
        return

    activity_names = [name for name, data in activity_data_tuples]
    activity_data_arrays = [data for name, data in activity_data_tuples]

    print("\nTestes de Normalidade (Kolmogorov-Smirnov) para cada atividade:")
    normality_results = {}
    use_kruskal = False

    for i, data in enumerate(activity_data_arrays):
        if len(data) >= 8:
            stat_ks, p_value_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            normality_results[activity_names[i]] = p_value_ks
            print(f"  Atividade '{activity_names[i]}' ({len(data)} amostras): p-value = {p_value_ks:.4f} {'(Não Normal)' if p_value_ks < 0.05 else '(Normal)'}")
            if p_value_ks < 0.05:
                use_kruskal = True
        else:
            print(f"  Atividade '{activity_names[i]}' ({len(data)} amostras): Amostra muito pequena para K-S test. Assumindo não normalidade para fins de teste de comparação.")
            normality_results[activity_names[i]] = 0.0
            use_kruskal = True


    if len(activity_data_arrays) < 2:
        print(f"\nNão há atividades suficientes ({len(activity_data_arrays)}) para realizar testes de comparação de médias/medianas.")
        return

    # 2. Escolher e 3. Realizar o teste estatístico apropriado
    if use_kruskal:
        print("\nUma ou mais distribuições não são normais ou têm amostras pequenas. Usando o teste Kruskal-Wallis (não-paramétrico).")
        stat, p_value = stats.kruskal(*activity_data_arrays)
        test_name = "Kruskal-Wallis H-test"
    else:
        print("\nAs distribuições parecem ser aproximadamente normais para todas as atividades. Usando ANOVA (paramétrico).")
        stat_levene, p_levene = stats.levene(*activity_data_arrays)
        print(f"  Teste de Levene para homogeneidade de variâncias: p-value = {p_levene:.4f}")
        if p_levene < 0.05:
            print("  ATENÇÃO: As variâncias entre os grupos são significativamente diferentes (p-value de Levene < 0.05). A ANOVA pode não ser robusta. Os resultados devem ser interpretados com cautela ou considerar testes mais avançados (ex: Welch's ANOVA).")

        stat, p_value = stats.f_oneway(*activity_data_arrays)
        test_name = "One-way ANOVA"

    print(f"\nResultado do {test_name} para a magnitude de {label}:")
    print(f"  Estatística: {stat:.4f}")
    print(f"  P-valor: {p_value:.4f}")

    # 4. Comentar os resultados
    print("\nComentário:")
    if p_value < 0.05:
        print(f"Com um p-valor de {p_value:.4f} (menor que 0.05), rejeitamos a hipótese nula de que os valores médios (ou medianas, no caso de Kruskal-Wallis) da magnitude de {label} são iguais em todas as atividades.")
        print(f"Isso indica que existe uma diferença estatisticamente significativa nos valores desta característica entre pelo menos algumas das atividades humanas.")
        print(f"Portanto, a magnitude de {label} demonstra ser uma característica promissora para diferenciar as atividades, sendo útil na pipeline de classificação de atividades humanas.")
    else:
        print(f"Com um p-valor de {p_value:.4f} (maior ou igual a 0.05), não há evidência estatística suficiente para rejeitar a hipótese nula de que os valores médios (ou medianas, no caso de Kruskal-Wallis) da magnitude de {label} são iguais em todas as atividades.")
        print(f"Isso sugere que a magnitude de {label}, por si só, pode não ser uma característica forte para distinguir entre as atividades analisadas, pois as suas médias/medianas não são significativamente diferentes.")
        print(f"Pode ser necessário combinar esta característica com outras, ou procurar características mais discriminativas, ou investigar interações complexas que não são capturadas por uma simples comparação de médias/medianas.")
