import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# =============================================================================
# ÉTAPE 0 : CONFIGURATION
# =============================================================================
INPUT_BRUTE_MATRIX_CSV = 'matrice_brute_valeurs_reelles.csv'
OUTPUT_REPORT_DIR = 'rapport_separabilite'
ALPHA_VALUES_TO_TEST = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
EPSILON = 1e-9

# =============================================================================
# ÉTAPE 1 : FONCTIONS DE PRÉPARATION ET DE CALCUL (identiques au script final)
# =============================================================================

def load_and_prepare_from_csv(csv_path: str):
    """
    Charge la matrice brute, la sépare, la parse, l'impute et la normalise.
    """
    print(f"--- Étape 1: Chargement et préparation depuis '{csv_path}' ---")
    try:
        df_full = pd.read_csv(csv_path).set_index('ID')
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{csv_path}' est introuvable.")
        return None, None
    
    p_expert_raw = df_full.loc['weight']
    df_taxa_raw = df_full.drop('weight')

    df_taxa_parsed = df_taxa_raw.copy()
    replacement_map = {'present': 1.0, 'absent': 0.0, 'inapplicable': 0.0, 'unknown': np.nan}
    
    def parse_numerical_value(value):
        if pd.isna(value): return np.nan
        s_value = str(value).strip().lower()
        try: return float(s_value)
        except ValueError: pass
        s_value = re.sub(r'[a-z"\'\s]+', '', s_value).replace(',', '.')
        if '-' in s_value:
            try:
                parts = [float(p) for p in s_value.split('-') if p]
                return np.mean(parts) if parts else np.nan
            except (ValueError, IndexError): return np.nan
        try:
            cleaned_num = re.sub(r'[^\d\.]', '', s_value)
            return float(cleaned_num) if cleaned_num else np.nan
        except ValueError: return np.nan

    for col in df_taxa_parsed.columns:
        df_taxa_parsed[col] = df_taxa_parsed[col].apply(lambda x: replacement_map.get(str(x).lower(), x)).apply(parse_numerical_value)

    numeric_cols = [col for col in df_taxa_parsed.columns if df_taxa_parsed[col].dropna().nunique() > 2]
    for col in df_taxa_parsed.columns:
        if df_taxa_parsed[col].isnull().any():
            if col in numeric_cols:
                df_taxa_parsed[col] = df_taxa_parsed[col].fillna(df_taxa_parsed[col].mean())
            else:
                df_taxa_parsed[col] = df_taxa_parsed[col].fillna(0)

    if numeric_cols:
        scaler = MinMaxScaler()
        df_taxa_parsed[numeric_cols] = scaler.fit_transform(df_taxa_parsed[numeric_cols])
        
    print("Données chargées, parsées, imputées et normalisées.")
    return df_taxa_parsed, p_expert_raw

def calculate_normalized_weights(features_df, p_expert_raw):
    """Calcule et normalise les poids de fréquence et experts."""
    print("--- Étape 2: Calcul et normalisation des poids ---")
    N = len(features_df)
    n_e = features_df.sum(axis=0)
    p_frequency = np.log(N / (n_e + 1))
    p_frequency.replace([np.inf, -np.inf], 0, inplace=True)

    p_expert = p_expert_raw.reindex(features_df.columns)
    if p_expert.isnull().any(): p_expert.fillna(p_expert.mean(), inplace=True)
    
    scaler = MinMaxScaler()
    p_freq_norm = pd.Series(scaler.fit_transform(p_frequency.values.reshape(-1, 1)).flatten(), index=p_frequency.index)
    p_expert_norm = pd.Series(scaler.fit_transform(p_expert.values.reshape(-1, 1)).flatten(), index=p_expert.index)
    
    print("Poids de fréquence et experts normalisés.")
    return p_freq_norm, p_expert_norm

def calculate_distance_matrix(data_matrix, weights_vector):
    """Calcule la matrice de distance Tanimoto pondérée."""
    def distance_function(u, v):
        dot = np.sum(weights_vector * u * v)
        norm_u = np.sum(weights_vector * u**2)
        norm_v = np.sum(weights_vector * v**2)
        denom = norm_u + norm_v - dot
        similarity = np.clip(dot / (denom + EPSILON), 0, 1)
        return 1.0 - similarity
    return pairwise_distances(data_matrix, metric=distance_function, n_jobs=-1)

# =============================================================================
# ÉTAPE 3 : FONCTION PRINCIPALE D'ANALYSE DE SÉPARABILITÉ
# =============================================================================

def analyze_global_separability():
    """
    Exécute l'analyse de séparabilité globale pour toutes les valeurs d'alpha
    et génère un rapport visuel.
    """
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    
    features_df, p_expert_raw = load_and_prepare_from_csv(INPUT_BRUTE_MATRIX_CSV)
    if features_df is None: return

    p_freq_norm, p_expert_norm = calculate_normalized_weights(features_df, p_expert_raw)

    # Extraire les genres pour chaque taxon
    taxa_names = features_df.index
    genres = [name.split(' ')[0] for name in taxa_names]

    separability_scores = []

    for alpha in ALPHA_VALUES_TO_TEST:
        print(f"\n==================== Analyse pour alpha = {alpha:.2f} ====================")
        
        # Calculer les poids et la matrice de distance pour cet alpha
        w = alpha * p_expert_norm + (1 - alpha) * p_freq_norm
        dist_matrix_np = calculate_distance_matrix(features_df.to_numpy(), w.to_numpy())
        df_distance = pd.DataFrame(dist_matrix_np, index=taxa_names, columns=taxa_names)

        # Calculer les distances intra- et inter-genre
        distances_intra_genre = []
        distances_inter_genre = []
        
        for i in range(len(taxa_names)):
            for j in range(i + 1, len(taxa_names)):
                distance = df_distance.iloc[i, j]
                if genres[i] == genres[j]:
                    distances_intra_genre.append(distance)
                else:
                    distances_inter_genre.append(distance)

        mean_intra = np.mean(distances_intra_genre) if distances_intra_genre else 0
        mean_inter = np.mean(distances_inter_genre) if distances_inter_genre else 0
        separability_score = mean_inter - mean_intra
        
        separability_scores.append({
            'alpha': alpha, 
            'separability': separability_score,
            'mean_intra_genre_dist': mean_intra,
            'mean_inter_genre_dist': mean_inter
        })
        print(f"  Score de séparabilité: {separability_score:.4f}")

    # --- Visualisation des Résultats ---
    results_df = pd.DataFrame(separability_scores).sort_values('alpha')

    optimal_alpha_df = results_df.loc[results_df['separability'].idxmax()]
    optimal_alpha = optimal_alpha_df['alpha']
    
    print("\n\n" + "="*50)
    print("      RAPPORT D'ANALYSE DE SÉPARABILITÉ GLOBALE")
    print("="*50)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print(f"\nALPHA OPTIMAL GLOBALEMENT : {optimal_alpha:.2f} (Score = {optimal_alpha_df['separability']:.4f})")
    print("="*50)

    # Créer le graphique
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    sns.lineplot(data=results_df, x='alpha', y='separability', marker='o', ax=ax1, color='red', label='Score de Séparabilité (Inter - Intra)')
    ax1.set_xlabel("Valeur de Alpha", fontsize=12)
    ax1.set_ylabel("Score de Séparabilité", color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=results_df, x='alpha', y='mean_intra_genre_dist', marker='s', ax=ax2, color='blue', linestyle='--', label='Dist. moyenne Intra-Genre')
    sns.lineplot(data=results_df, x='alpha', y='mean_inter_genre_dist', marker='^', ax=ax2, color='green', linestyle='--', label='Dist. moyenne Inter-Genre')
    ax2.set_ylabel("Distance Moyenne", fontsize=12)
    
    plt.axvline(x=optimal_alpha, color='black', linestyle=':', label=f'Alpha Optimal ({optimal_alpha:.2f})')
    
    fig.suptitle("Analyse de la Séparabilité Globale en fonction d'Alpha", fontsize=16)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    plt.xticks(ALPHA_VALUES_TO_TEST, rotation=45)

    # Sauvegarder le graphique
    chart_path = os.path.join(OUTPUT_REPORT_DIR, 'analyse_separabilite_globale.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraphique de séparabilité sauvegardé dans : '{chart_path}'")


if __name__ == "__main__":
    analyze_global_separability()