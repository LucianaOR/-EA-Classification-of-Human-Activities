import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def is_empty_like(x):
    if x is None:
        return True
    try:
        if hasattr(x, 'empty'):
            return bool(x.empty)
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.size == 0
        if isinstance(x, (list, tuple, set, dict)):
            return len(x) == 0
        try:
            return len(x) == 0
        except Exception:
            return False
    except Exception:
        return False

# -------------------------------
# 3.1 - Boxplot por atividade
# -------------------------------
def boxplot_by_activity(df, variable, title):
    df_valid = df[df[variable].notna() & df["activity"].notna()]
    if is_empty_like(df_valid):
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center", fontsize=12)
        plt.axis("off")
        return

    sns.boxplot(
        data=df_valid,
        x="activity",
        y=variable,
        palette="coolwarm",
        legend=False,
        fliersize=3
    )

    plt.title(title)
    plt.xlabel("Atividade (1–16)")
    plt.ylabel(variable)
    plt.grid(True, linestyle="--", alpha=0.5)

# -------------------------------
# 3.2 - Densidade de outliers
# Detectar outliers com IQR/Tukey
# -------------------------------
def iqr_outliers(array):
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (array < lower_bound) | (array > upper_bound)

def outlier_density_iqr(df, variable):
    activities = sorted(df["activity"].unique())
    densities = {}
    for act in activities:
        act_values = df[df["activity"] == act][variable].values
        outliers = iqr_outliers(act_values)
        density = np.sum(outliers) / len(act_values) * 100
        densities[act] = density
        print(f"Atividade {act}: {density:.2f}% outliers")
    return densities

def plot_outliers_iqr(df, variable, title="Outliers"):
    activities = sorted(df["activity"].unique())
    plt.figure(figsize=(12, 6))

    for act in activities:
        subset = df[df["activity"] == act]
        values = subset[variable].values
        outliers = iqr_outliers(values)
        colors = ["red" if flag else "blue" for flag in outliers]

        plt.scatter(
            [act] * len(values),
            values,
            c=colors,
            s=10,
            alpha=0.7
        )

    plt.title(f"{title} ({variable}) - IQR")
    plt.xlabel("Atividade")
    plt.ylabel(variable)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(activities)
    plt.show()


# -------------------------------
# 3.3 / 3.4 / 3.5 - Z-Score Outliers e Comparação
# -------------------------------

def zscore_outliers(array, k=3):
    array = np.array(array)
    array = array[~np.isnan(array)] 
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        return np.zeros_like(array, dtype=bool)
    z_scores = np.abs((array - mean) / std)
    return z_scores > k


def compare_zscores(df, variable, k_list=[3, 3.5, 4]):
    activities = sorted(df["activity"].unique())

    for k in k_list:
        print(f"\n--- Z-Score k={k} ---")
        plt.figure(figsize=(12, 6))
        total_outliers = 0
        total_points = 0

        for act in activities:
            subset = df[df["activity"] == act]
            values = subset[variable].dropna().values

            if len(values) == 0:
                continue

            outliers = zscore_outliers(values, k)
            density = np.sum(outliers) / len(values) * 100
            total_outliers += np.sum(outliers)
            total_points += len(values)

            colors = ["red" if flag else "blue" for flag in outliers]
            plt.scatter([act] * len(values), values, c=colors, s=10, alpha=0.7)

            print(f"Atividade {act}: {density:.2f}% outliers (k={k})")

        plt.title(f"{variable} - Z-Score (k={k})")
        plt.xlabel("Atividade")
        plt.ylabel(variable)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.show()

        if total_points > 0:
            overall_density = total_outliers / total_points * 100
            print(f"Densidade total de outliers ({variable}, k={k}): {overall_density:.2f}%")

