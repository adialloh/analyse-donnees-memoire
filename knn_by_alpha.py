import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator # Nous le gardons pour comparer
from collections import Counter

# =============================================================================
# ÉTAPE 0 : CONFIGURATION
# =============================================================================
INPUT_BRUTE_MATRIX_CSV = 'matrice_brute_valeurs_reelles.csv'
OPTIMAL_ALPHA = 0.70 # Votre alpha optimisé
EPSILON = 1e-9

# =============================================================================
# CLASSE PRINCIPALE POUR L'IDENTIFICATION (AVEC MÉTHODE DE COUDE AMÉLIORÉE)
# =============================================================================

class SmartKnnIdentifier:
    """
    Un classifieur k-NN qui détermine dynamiquement k en utilisant une méthode
    de détection de coude robuste (méthode du triangle).
    """

    def __init__(self, brute_matrix_path: str, alpha: float):
        # ... (Le code de __init__, _load_and_prepare_data, _calculate_hybrid_weights, 
        #      et _calculate_tanimoto_distance est exactement le même que dans la réponse précédente) ...
        print("Initialisation de l'identificateur intelligent...")
        self.alpha = alpha
        self.features_df, self.expert_weights_raw = self._load_and_prepare_data(brute_matrix_path)
        if self.features_df is None:
            raise FileNotFoundError(f"Le fichier '{brute_matrix_path}' est introuvable.")
        self.hybrid_weights = self._calculate_hybrid_weights()
        print(f"Poids calculés avec succès pour alpha = {self.alpha}. Identificateur prêt.")

    # ... (Copiez ici les méthodes _load_and_prepare_data, _calculate_hybrid_weights, _calculate_tanimoto_distance) ...
    def _load_and_prepare_data(self, csv_path: str):
        # ... (code de la fonction) ...
        try:
            df_full = pd.read_csv(csv_path).set_index('ID')
        except FileNotFoundError:
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
            
        return df_taxa_parsed, p_expert_raw

    def _calculate_hybrid_weights(self):
        # ... (code de la fonction) ...
        N = len(self.features_df)
        n_e = self.features_df.sum(axis=0)
        p_frequency = np.log(N / (n_e + 1))
        p_frequency.replace([np.inf, -np.inf], 0, inplace=True)
        p_expert = self.expert_weights_raw.reindex(self.features_df.columns)
        if p_expert.isnull().any():
            p_expert.fillna(p_expert.mean(), inplace=True)
        scaler = MinMaxScaler()
        p_freq_norm = pd.Series(scaler.fit_transform(p_frequency.values.reshape(-1, 1)).flatten(), index=p_frequency.index)
        p_expert_norm = pd.Series(scaler.fit_transform(p_expert.values.reshape(-1, 1)).flatten(), index=p_expert.index)
        return self.alpha * p_expert_norm + (1 - self.alpha) * p_freq_norm
        
    def _calculate_tanimoto_distance(self, u, v):
        # ... (code de la fonction) ...
        w = self.hybrid_weights.values
        dot = np.sum(w * u * v)
        norm_u = np.sum(w * u**2)
        norm_v = np.sum(w * v**2)
        denom = norm_u + norm_v - dot
        similarity = np.clip(dot / (denom + EPSILON), 0, 1)
        return 1.0 - similarity
        
    def _find_optimal_k(self, distances: np.ndarray, max_k_ratio=0.25, k_default=7):
        """
        Trouve le k optimal en utilisant la méthode de la distance à la ligne (triangle).
        """
        # Limiter la recherche à une portion raisonnable de la courbe pour éviter les faux coudes
        max_k = int(len(distances) * max_k_ratio)
        if max_k < 5: max_k = min(len(distances), 5) # S'assurer qu'on a un minimum de points
        
        x_vals = np.arange(1, len(distances) + 1)
        y_vals = distances
        
        # Ne considérer que la plage de recherche
        search_x = x_vals[:max_k]
        search_y = y_vals[:max_k]

        if len(search_x) < 3:
            return k_default # Pas assez de points pour la méthode du triangle

        # Normaliser les points pour que les échelles x et y n'affectent pas la distance
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_norm = scaler_x.fit_transform(search_x.reshape(-1, 1)).flatten()
        y_norm = scaler_y.fit_transform(search_y.reshape(-1, 1)).flatten()

        # Points de début et de fin de la ligne de référence
        start_point = np.array([x_norm[0], y_norm[0]])
        end_point = np.array([x_norm[-1], y_norm[-1]])

        # Calculer la distance de chaque point à la ligne
        line_vec = end_point - start_point
        line_vec_norm = np.linalg.norm(line_vec)
        if line_vec_norm == 0: return k_default

        vec_from_start = np.array([x_norm, y_norm]).T - start_point
        
        # Utiliser le produit vectoriel pour trouver la distance perpendiculaire
        cross_product = np.cross(vec_from_start, line_vec / line_vec_norm)
        distances_to_line = np.abs(cross_product)
        
        # L'index du point le plus distant est notre coude
        elbow_index = np.argmax(distances_to_line)
        
        # Retourner la valeur de k correspondante (l'index + 1)
        optimal_k = search_x[elbow_index]
        
        return optimal_k

    def identify(self, unknown_specimen: dict, plot_elbow: bool = True):
        """
        Identifie un spécimen en utilisant k déterminé par la méthode du triangle.
        """
        print(f"\n--- Début de l'identification pour le spécimen inconnu ---")
        
        unknown_series = pd.Series(unknown_specimen)
        unknown_vector = unknown_series.reindex(self.features_df.columns).fillna(0).values
        
        distances = [{'taxon': index, 'distance': self._calculate_tanimoto_distance(unknown_vector, row.values)}
                     for index, row in self.features_df.iterrows()]
        df_distances = pd.DataFrame(distances).sort_values('distance').reset_index(drop=True)
        
        # Déterminer k avec notre nouvelle méthode fiable
        k = self._find_optimal_k(df_distances['distance'].values)
        print(f"Méthode du triangle a déterminé k = {k}")

        if plot_elbow:
            plt.figure(figsize=(10, 6))
            x_vals_full = df_distances.index.values + 1
            y_vals_full = df_distances['distance'].values
            sns.lineplot(x=x_vals_full, y=y_vals_full, marker='o', label="Distances triées")
            plt.axvline(x=k, color='red', linestyle='--', label=f'Coude fiable trouvé à k={k}')
            plt.title("Détermination de k par la Méthode du Triangle")
            plt.xlabel("Rang du Voisin")
            plt.ylabel("Distance de Tanimoto Pondérée")
            plt.legend()
            plt.show()

        # ... (Le reste de la fonction : vote, résultat, etc. est identique) ...
        nearest_neighbors = df_distances.head(k)
        neighbor_genres = [name.split(' ')[0] for name in nearest_neighbors['taxon']]
        vote_counts = Counter(neighbor_genres)
        
        if not vote_counts:
            # ...
            predicted_genre, vote_count, reasoning = "Inconnue", 0, "Impossible de trouver des voisins."
        else:
            predicted_genre, vote_count = vote_counts.most_common(1)[0]
            reasoning = f"L'identification est '{predicted_genre}' basé sur un vote majoritaire de {vote_count} sur {k} voisins (k trouvé automatiquement)."

        return {
            "identification": predicted_genre, "k_found": k, "vote_details": dict(vote_counts),
            "neighbors": nearest_neighbors.to_dict('records'), "reasoning": reasoning
        }

# =============================================================================
# ÉTAPE 5 : EXÉCUTION DE L'EXEMPLE
# =============================================================================
if __name__ == "__main__":
    try:
        identifier = SmartKnnIdentifier(INPUT_BRUTE_MATRIX_CSV, alpha=OPTIMAL_ALPHA)
        
        TAXON_DE_REFERENCE = 'Albizia adianthifolia' # Ou un autre taxon de votre choix
        if TAXON_DE_REFERENCE not in identifier.features_df.index:
            raise ValueError(f"Le taxon de référence '{TAXON_DE_REFERENCE}' n'existe pas dans la matrice.")
        
        print(f"\nUtilisation de '{TAXON_DE_REFERENCE}' comme base pour le spécimen inconnu.")
        specimen_inconnu = identifier.features_df.loc[TAXON_DE_REFERENCE].to_dict()
        
        identification_result = identifier.identify(specimen_inconnu)
        
        # ... (Rapport d'affichage final identique à la version précédente) ...
        print("\n\n" + "="*50)
        print("      RÉSULTAT FINAL DE L'IDENTIFICATION")
        print("="*50)
        print(f"Spécimen de test basé sur      : {TAXON_DE_REFERENCE}")
        print(f"Identification la plus probable : Genre '{identification_result['identification']}'")
        print(f"K déterminé automatiquement    : {identification_result['k_found']}")
        print(f"Détails du vote                : {identification_result['vote_details']}")
        print(f"Raisonnement                   : {identification_result['reasoning']}")
        print("\nPlus proches voisins :")
        for neighbor in identification_result['neighbors']:
            is_self = " (Lui-même)" if neighbor['taxon'] == TAXON_DE_REFERENCE else ""
            print(f"  - {neighbor['taxon']:<30} (Distance: {neighbor['distance']:.4f}){is_self}")
        
        predicted_genre = identification_result['identification']
        true_genre = TAXON_DE_REFERENCE.split(' ')[0]
        if predicted_genre == true_genre:
            print(f"\nConclusion : SUCCÈS ! Le genre prédit ('{predicted_genre}') correspond au genre du taxon de référence.")
        else:
            print(f"\nConclusion : ÉCHEC. Le genre prédit ('{predicted_genre}') ne correspond pas au genre du taxon de référence ('{true_genre}').")
        print("="*50)

    except Exception as e:
        # ... (Gestion des erreurs identique) ...
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc()