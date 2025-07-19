import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator # Gardé pour comparaison visuelle

# =============================================================================
# ÉTAPE 0 : CONFIGURATION
# =============================================================================
DB_FILE = 'mydb.db'
OUTPUT_DIRECTORY = 'alpha_sensitivity_analysis_sql_corrected'
ALPHA_VALUES_TO_TEST = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0]
TARGET_TAXON = 'Albizia adianthifolia' # Assurez-vous que ce taxon est bien dans votre DB
EPSILON = 1e-9

# =============================================================================
# ÉTAPE 1 : CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
def load_and_prepare_data_from_sql_pivot(db_file):
    """Charge les données via la requête SQL pivot, les nettoie, impute et normalise."""
    print("--- Étape 1: Chargement et préparation des données via SQL Pivot ---")
    
    # ATTENTION : Cette requête est performante mais difficile à maintenir.
    # Tout ajout de descripteur dans la DB nécessite une modification manuelle ici.
    sql_query_matrix = """
    SELECT
        t.id AS taxon_id, t.scientific_name,
        MAX(CASE WHEN o.descriptor_node_id = 3   THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Introduite_3',
        MAX(CASE WHEN o.descriptor_node_id = 9   THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Arbre_9',
        MAX(CASE WHEN o.descriptor_node_id = 10  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Arbuste_10',
        MAX(CASE WHEN o.descriptor_node_id = 12  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Liane_12',
        MAX(CASE WHEN o.descriptor_node_id = 13  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Herbe_13',
        MAX(CASE WHEN o.descriptor_node_id = 17  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Vivace_17',
        MAX(CASE WHEN o.descriptor_node_id = 20  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Caduque_20',
        MAX(CASE WHEN o.descriptor_node_id = 54  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Presence_exsudat_54',
        MAX(CASE WHEN o.descriptor_node_id = 66  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Phyllo_Alterne_66',
        MAX(CASE WHEN o.descriptor_node_id = 72  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Limbe_Composee_72',
        MAX(CASE WHEN o.descriptor_node_id = 75  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Composee_Bipennee_75',
        MAX(CASE WHEN o.descriptor_node_id = 82  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Presence_stipules_82',
        MAX(CASE WHEN o.descriptor_node_id = 87  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Presence_pulvinus_87',
        MAX(CASE WHEN o.descriptor_node_id = 90  THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Marge_Entiere_90',
        MAX(CASE WHEN o.descriptor_node_id = 140 THEN CASE o.observed_value WHEN 'present' THEN '1' WHEN 'absent' THEN '0' WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE NULL END END) AS 'Fruit_Baie_140',
        MAX(CASE WHEN o.descriptor_node_id = 79 THEN CASE o.observed_value WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE CAST(CASE WHEN INSTR(o.observed_value, '-') > 0 THEN (CAST(SUBSTR(o.observed_value, 1, INSTR(o.observed_value, '-') - 1) AS REAL) + CAST(SUBSTR(o.observed_value, INSTR(o.observed_value, '-') + 1) AS REAL)) / 2.0 ELSE CAST(REPLACE(REPLACE(o.observed_value, '>', ''), '<', '') AS REAL) END AS TEXT) END END) AS 'longueur_feuille_cm_79',
        MAX(CASE WHEN o.descriptor_node_id = 80 THEN CASE o.observed_value WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE CAST(CASE WHEN INSTR(o.observed_value, '-') > 0 THEN (CAST(SUBSTR(o.observed_value, 1, INSTR(o.observed_value, '-') - 1) AS REAL) + CAST(SUBSTR(o.observed_value, INSTR(o.observed_value, '-') + 1) AS REAL)) / 2.0 ELSE CAST(o.observed_value AS REAL) END AS TEXT) END END) AS 'largeur_feuille_cm_80',
        MAX(CASE WHEN o.descriptor_node_id = 81 THEN CASE o.observed_value WHEN 'inapplicable' THEN '0' WHEN 'unknown' THEN 'nan' ELSE CAST(CASE WHEN INSTR(o.observed_value, '-') > 0 THEN (CAST(SUBSTR(o.observed_value, 1, INSTR(o.observed_value, '-') - 1) AS REAL) + CAST(SUBSTR(o.observed_value, INSTR(o.observed_value, '-') + 1) AS REAL)) / 2.0 ELSE CAST(o.observed_value AS REAL) END AS TEXT) END END) AS 'nombre_folioles_81'
    FROM taxa t
    LEFT JOIN observations o ON t.id = o.taxon_id
    WHERE t.rank_id = 25
    GROUP BY t.id, t.scientific_name
    ORDER BY t.id;
    """
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(sql_query_matrix, conn)
    conn.close()

    df.replace('nan', np.nan, inplace=True)
    df.set_index('scientific_name', inplace=True)
    features_df = df.drop(columns=['taxon_id'])
    
    features_df = features_df.astype(float)

    numeric_cols = [col for col in features_df.columns if col.startswith(('longueur', 'largeur', 'nombre'))]
    binary_cols = [col for col in features_df.columns if col not in numeric_cols]
    
    if binary_cols and features_df[binary_cols].isnull().values.any():
        imputer_binary = SimpleImputer(strategy='median')
        features_df[binary_cols] = imputer_binary.fit_transform(features_df[binary_cols])
    
    if numeric_cols and features_df[numeric_cols].isnull().values.any():
        imputer_numeric = SimpleImputer(strategy='mean')
        features_df[numeric_cols] = imputer_numeric.fit_transform(features_df[numeric_cols])

    if numeric_cols:
        scaler = MinMaxScaler()
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

    print("Données chargées, imputées et normalisées.")
    return features_df

# =============================================================================
# ÉTAPE 2 : CALCUL DES POIDS
# =============================================================================
def calculate_normalized_weights(features_df, db_file):
    """Calcule et normalise les poids de fréquence et experts."""
    print("--- Étape 2: Calcul et normalisation des poids ---")
    N = len(features_df)
    n_e = features_df.sum(axis=0)
    p_frequency = np.log(N / (n_e + 1))
    p_frequency.replace([np.inf, -np.inf], 0, inplace=True)

    # Créer une liste d'IDs à partir des noms de colonnes pour rendre la requête dynamique
    descriptor_ids = [int(col.split('_')[-1]) for col in features_df.columns]

    conn = sqlite3.connect(db_file)
    # Rendre la requête des poids plus maintenable
    query_weights = f"SELECT id, discriminant_power_on20 FROM nodes WHERE id IN ({','.join(map(str, descriptor_ids))})"
    expert_weights_db = pd.read_sql_query(query_weights, conn)
    conn.close()
    
    expert_weights_map = pd.Series(expert_weights_db['discriminant_power_on20'].values, index=expert_weights_db['id'])
    p_expert = pd.Series(index=features_df.columns, dtype=float)
    for col in p_expert.index:
        descriptor_id = int(col.split('_')[-1])
        p_expert[col] = expert_weights_map.get(descriptor_id)
    
    imputer_expert = SimpleImputer(strategy='mean')
    p_expert_imputed = imputer_expert.fit_transform(p_expert.values.reshape(-1, 1)).flatten()
    p_expert = pd.Series(p_expert_imputed, index=p_expert.index)
    
    scaler = MinMaxScaler()
    p_freq_norm = pd.Series(scaler.fit_transform(p_frequency.values.reshape(-1, 1)).flatten(), index=p_frequency.index)
    p_expert_norm = pd.Series(scaler.fit_transform(p_expert.values.reshape(-1, 1)).flatten(), index=p_expert.index)
    
    print("Poids de fréquence et experts normalisés.")
    return p_freq_norm, p_expert_norm

# =============================================================================
# ÉTAPE 3 : CALCUL DE DISTANCE ET DÉTECTION DE COUDE
# =============================================================================
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
    if max_k < 5: max_k = min(len(distances), 5)
    
    search_x, search_y = np.arange(1, max_k + 1), distances[:max_k]
    if len(search_x) < 3: return k_default

    x_norm = (search_x - search_x.min()) / (search_x.max() - search_x.min())
    y_norm = (search_y - search_y.min()) / (search_y.max() - search_y.min())

    start_point, end_point = np.array([x_norm[0], y_norm[0]]), np.array([x_norm[-1], y_norm[-1]])
    line_vec = end_point - start_point
    line_vec_norm = np.linalg.norm(line_vec)
    if line_vec_norm == 0: return k_default

    vec_from_start = np.array([x_norm, y_norm]).T - start_point
    cross_product = np.abs(np.cross(vec_from_start, line_vec / line_vec_norm))
    return search_x[np.argmax(cross_product)]

# =============================================================================
# ÉTAPE 4 : FONCTION PRINCIPALE D'ANALYSE
# =============================================================================
def main():
    """Exécute l'analyse de sensibilité de bout en bout."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    features_df = load_and_prepare_data_from_sql_pivot(DB_FILE)
    if features_df is None or TARGET_TAXON not in features_df.index:
        print(f"ERREUR : Le taxon cible '{TARGET_TAXON}' est introuvable ou les données n'ont pu être chargées.")
        return

    p_freq_norm, p_expert_norm = calculate_normalized_weights(features_df, DB_FILE)
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
        
        output_path = os.path.join(OUTPUT_DIRECTORY, f'distances_alpha_{alpha:.2f}.csv')
        distance_matrix.to_csv(output_path, float_format='%.6f')
        print(f"Matrice de distance sauvegardée.")

        distances_to_target = distance_matrix[TARGET_TAXON].drop(TARGET_TAXON).sort_values()
        
        # CORRECTION MAJEURE : Utiliser la méthode du triangle
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
        
        print(f"Nombre de voisins déterminé par la méthode du triangle : N = {n_neighbors}")

    for i in range(num_alphas, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Analyse de l'Influence d'Alpha sur la Distance pour '{TARGET_TAXON}'", fontsize=18)
    
    composite_plot_path = os.path.join(OUTPUT_DIRECTORY, 'analyse_alpha_subplots.png')
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
    main()