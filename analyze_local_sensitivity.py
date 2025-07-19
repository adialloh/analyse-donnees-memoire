import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from collections import Counter

# =============================================================================
# ÉTAPE 0 : CONFIGURATION
# =============================================================================
INPUT_BRUTE_MATRIX_CSV = 'matrice_brute_valeurs_reelles.csv'
OUTPUT_DIRECTORY = 'analyse_sensibilite_locale'
TARGET_TAXON = 'Albizia adianthifolia' # Le taxon de référence pour l'analyse
ALPHA_VALUES_TO_TEST = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
EPSILON = 1e-9

# =============================================================================
# ÉTAPE 1 : FONCTIONS DE PRÉPARATION ET DE CALCUL (identiques aux scripts précédents)
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

def find_optimal_k_triangle_method(distances: np.ndarray, max_k_ratio=0.25, k_default=7):
    """Trouve le k optimal en utilisant la méthode du triangle."""
    max_k = int(len(distances) * max_k_ratio)
    if max_k < 5: max_k = min(len(distances), 10) # Augmenter la taille min de recherche
    
    search_x, search_y = np.arange(1, max_k + 1), distances[:max_k]
    if len(search_x) < 3: return k_default

    x_norm = (search_x - search_x.min()) / (search_x.max() - search_x.min() + EPSILON)
    y_norm = (search_y - search_y.min()) / (search_y.max() - search_y.min() + EPSILON)

    start_point, end_point = np.array([x_norm[0], y_norm[0]]), np.array([x_norm[-1], y_norm[-1]])
    line_vec = end_point - start_point
    line_vec_norm = np.linalg.norm(line_vec)
    if line_vec_norm == 0: return k_default

    all_points = np.column_stack((x_norm, y_norm))
    vec_from_start = all_points - start_point
    distances_to_line = np.abs(np.cross(vec_from_start, line_vec / line_vec_norm))
    
    return search_x[np.argmax(distances_to_line)]

# =============================================================================
# ÉTAPE 4 : FONCTION PRINCIPALE D'ANALYSE DE SENSIBILITÉ LOCALE
# =============================================================================
def analyze_local_sensitivity():
    """
    Exécute l'analyse de sensibilité locale en se concentrant sur un taxon cible
    et en générant un rapport visuel et textuel.
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    features_df, p_expert_raw = load_and_prepare_from_csv(INPUT_BRUTE_MATRIX_CSV)
    if features_df is None: return
    
    if TARGET_TAXON not in features_df.index:
        print(f"ERREUR : Le taxon cible '{TARGET_TAXON}' n'a pas été trouvé dans la matrice.")
        return

    p_freq_norm, p_expert_norm = calculate_normalized_weights(features_df, p_expert_raw)

    summary_results = {}
    
    num_alphas = len(ALPHA_VALUES_TO_TEST)
    n_cols = 3
    n_rows = int(np.ceil(num_alphas / n_cols))
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5), squeeze=False, constrained_layout=True)
    axes = axes.flatten()
    sns.set_theme(style="whitegrid", palette="viridis")

    for idx, alpha in enumerate(ALPHA_VALUES_TO_TEST):
        print(f"\n==================== TRAITEMENT POUR ALPHA = {alpha:.2f} ====================")
        
        w = alpha * p_expert_norm + (1 - alpha) * p_freq_norm
        data_matrix = features_df.to_numpy()
        weights_vector = w.reindex(features_df.columns).to_numpy()
        
        dist_matrix_np = calculate_distance_matrix(data_matrix, weights_vector)
        distance_matrix = pd.DataFrame(dist_matrix_np, index=features_df.index, columns=features_df.index)
        
        distances_to_target = distance_matrix[TARGET_TAXON].drop(TARGET_TAXON).sort_values()
        
        n_neighbors = find_optimal_k_triangle_method(distances_to_target.values)
        
        summary_results[alpha] = {'N': n_neighbors, 'neighbors': distances_to_target.head(n_neighbors)}

        ax = axes[idx]
        x_vals, y_vals = np.arange(1, len(distances_to_target) + 1), distances_to_target.values
        sns.lineplot(x=x_vals, y=y_vals, marker='o', ax=ax, label="Distances")
        ax.axvline(x=n_neighbors, color=sns.color_palette("Set2")[1], linestyle='--', label=f'Coude à N={n_neighbors}')
        
        ax.set_title(f"Alpha = {alpha:.2f}")
        ax.set_xlabel("Rang du Voisin")
        ax.set_ylabel("Distance")
        ax.legend()
        
        print(f"Nombre de voisins déterminé : N = {n_neighbors}")

    for i in range(num_alphas, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Analyse de l'Influence d'Alpha sur le Voisinage de '{TARGET_TAXON}'", fontsize=18)
    
    composite_plot_path = os.path.join(OUTPUT_DIRECTORY, 'analyse_alpha_locale_subplots.png')
    fig.savefig(composite_plot_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique composite sauvegardé dans : {composite_plot_path}")

    # Rapport de synthèse
    print("\n\n" + "="*70)
    print(f" RAPPORT DE SYNTHÈSE : INFLUENCE D'ALPHA SUR LES PLUS PROCHES VOISINS")
    print(f"               Taxon Cible : {TARGET_TAXON}")
    print("="*70)
    for alpha, result in summary_results.items():
        n, neighbors = result['N'], result['neighbors']
        print(f"\n----- Alpha = {alpha:.2f} (N déterminé par le coude = {n}) -----")
        if alpha == 0.0: print("(100% Poids de Fréquence)")
        elif alpha == 1.0: print("(100% Poids Expert)")
        else: print(f"({(1-alpha)*100:.0f}% Fréquence / {alpha*100:.0f}% Expert)")
        print(neighbors.to_string(header=['Distance']))

if __name__ == "__main__":
    analyze_local_sensitivity()