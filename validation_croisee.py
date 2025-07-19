import pandas as pd
import numpy as np
import os
import re
from collections import Counter
import random

# =============================================================================
# ÉTAPE 0 : CONFIGURATION
# =============================================================================
INPUT_BRUTE_MATRIX_CSV = 'matrice_brute_valeurs_reelles.csv'
# Vos hyperparamètres optimaux validés
OPTIMAL_ALPHA = 0.70
EPSILON = 1e-9

# =============================================================================
# CLASSE D'IDENTIFICATION FINALE ET ROBUSTE
# =============================================================================

class FinalKnnIdentifier:
    """
    Un classifieur k-NN qui applique une métrique de distance de Tanimoto pondérée.
    Cette classe prépare les données et les poids pour l'identification.
    """

    def __init__(self, brute_matrix_path: str, alpha: float):
        """
        Initialise l'identificateur en chargeant et préparant toutes les données.
        """
        print("Initialisation de l'identificateur final...")
        self.alpha = alpha
        # Charger et préparer les données de référence
        self.features_df, self.expert_weights_raw = self._load_and_prepare_data(brute_matrix_path)
        if self.features_df is None:
            raise FileNotFoundError(f"Le fichier '{brute_matrix_path}' est introuvable.")
        # Calculer les poids qui seront utilisés pour toutes les identifications
        self.hybrid_weights = self._calculate_hybrid_weights()
        print(f"Poids calculés avec succès pour alpha = {self.alpha}. Identificateur prêt.")

    def _load_and_prepare_data(self, csv_path: str):
        """
        Charge la matrice brute depuis un CSV, la sépare, la parse, l'impute et la normalise.
        """
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
                except (ValueError, IndexError):
                    return np.nan
            try:
                cleaned_num = re.sub(r'[^\d\.]', '', s_value)
                return float(cleaned_num) if cleaned_num else np.nan
            except ValueError:
                return np.nan

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
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df_taxa_parsed[numeric_cols] = scaler.fit_transform(df_taxa_parsed[numeric_cols])
            
        return df_taxa_parsed, p_expert_raw

    def _calculate_hybrid_weights(self):
        """
        Calcule et normalise les poids de fréquence et experts, puis les combine avec alpha.
        """
        from sklearn.preprocessing import MinMaxScaler
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
        """
        Calcule une seule distance de Tanimoto pondérée.
        """
        w = self.hybrid_weights.values
        dot = np.sum(w * u * v)
        norm_u = np.sum(w * u**2)
        norm_v = np.sum(w * v**2)
        denom = norm_u + norm_v - dot
        similarity = np.clip(dot / (denom + EPSILON), 0, 1)
        return 1.0 - similarity

# =============================================================================
# SCRIPT DE VALIDATION AUTOMATISÉE
# =============================================================================

def get_ranked_list_for_validation(identifier: FinalKnnIdentifier, unknown_specimen: dict):
    """
    Calcule les distances d'un spécimen à tous les taxons de la base et retourne la liste classée.
    Le taxon cible n'est PAS retiré de la base de données de référence.
    """
    unknown_vector = pd.Series(unknown_specimen).reindex(identifier.features_df.columns).fillna(0).values
    
    distances = [{'taxon': index, 'distance': identifier._calculate_tanimoto_distance(unknown_vector, row.values)}
                 for index, row in identifier.features_df.iterrows()]
                 
    # Mettre le nom du taxon comme index pour une recherche facile
    return pd.DataFrame(distances).set_index('taxon').sort_values('distance')

def find_rank(ranked_list_df: pd.DataFrame, target_taxon: str):
    """
    Trouve le rang (position, commençant à 1) du taxon cible dans une liste classée.
    """
    try:
        # La position dans la liste triée est son rang
        rank = ranked_list_df.index.get_loc(target_taxon) + 1
        return rank
    except KeyError:
        # Si le taxon n'est pas trouvé (ne devrait pas arriver avec ce protocole)
        return float('inf')

def run_experimental_validation(identifier: FinalKnnIdentifier):
    """
    Exécute le protocole de validation complet (parfait, partiel, bruit)
    sur l'ensemble de la base de données et calcule les scores Top-n.
    """
    print("\n" + "="*50)
    print("      DÉBUT DE LA VALIDATION EXPÉRIMENTALE (Protocole Final)")
    print("="*50)
    
    all_taxa = identifier.features_df.index.tolist()
    results = []

    for i, target_taxon in enumerate(all_taxa):
        print(f"Validation du taxon {i+1}/{len(all_taxa)} : {target_taxon}...")
        
        original_profile = identifier.features_df.loc[target_taxon].to_dict()

        # --- Test 1: Données parfaites ---
        ranked_list_perfect = get_ranked_list_for_validation(identifier, original_profile)
        
        # --- Test 2: Données partielles (50% des descripteurs non nuls sont masqués) ---
        partial_profile = original_profile.copy()
        non_zero_keys = [k for k, v in partial_profile.items() if v != 0]
        # S'assurer qu'il y a au moins 2 descripteurs à masquer
        if len(non_zero_keys) > 1:
            keys_to_remove = random.sample(non_zero_keys, k=len(non_zero_keys) // 2)
            for key in keys_to_remove:
                del partial_profile[key] # Simule un descripteur non renseigné
        ranked_list_partial = get_ranked_list_for_validation(identifier, partial_profile)

        # --- Test 3: Données bruitées (2 erreurs introduites) ---
        noisy_profile = original_profile.copy()
        keys_to_noise = list(noisy_profile.keys())
        if len(keys_to_noise) >= 2:
            noise_keys = random.sample(keys_to_noise, k=2)
            for key in noise_keys:
                # Inverser la valeur binaire ou ajouter du bruit à la valeur numérique
                if noisy_profile[key] in [0, 1]:
                    noisy_profile[key] = 1 - noisy_profile[key]
                else: # Numérique
                    noisy_profile[key] *= (1 + random.uniform(-0.2, 0.2)) # Bruit de +/- 20%
        ranked_list_noisy = get_ranked_list_for_validation(identifier, noisy_profile)

        # --- Stocker les rangs trouvés pour chaque test ---
        results.append({
            'taxon': target_taxon,
            'rank_perfect': find_rank(ranked_list_perfect, target_taxon),
            'rank_partial': find_rank(ranked_list_partial, target_taxon),
            'rank_noisy': find_rank(ranked_list_noisy, target_taxon)
        })

    # --- Calculer et afficher les scores finaux ---
    results_df = pd.DataFrame(results)
    
    print("\n\n" + "="*50)
    print("      RÉSULTATS DE LA VALIDATION")
    print("="*50)
    
    for test_type in ['perfect', 'partial', 'noisy']:
        col_name = f'rank_{test_type}'
        print(f"\n--- Performance pour le test '{test_type.upper()}' ---")
        top1 = (results_df[col_name] == 1).mean() * 100
        top5 = (results_df[col_name] <= 5).mean() * 100
        top10 = (results_df[col_name] <= 10).mean() * 100
        print(f"  Taux de réussite Top-1  : {top1:.2f}%")
        print(f"  Taux de réussite Top-5  : {top5:.2f}%")
        print(f"  Taux de réussite Top-10 : {top10:.2f}%")
        
    print("="*50)
    return results_df

# =============================================================================
# EXÉCUTION DU SCRIPT
# =============================================================================
if __name__ == "__main__":
    try:
        # 1. Initialiser le système
        identifier = FinalKnnIdentifier(INPUT_BRUTE_MATRIX_CSV, alpha=OPTIMAL_ALPHA)
        
        # 2. Lancer la validation expérimentale complète
        validation_results = run_experimental_validation(identifier)
        
        # 3. (Optionnel) Sauvegarder les résultats détaillés de la validation
        validation_results.to_csv('resultats_validation_detailles.csv', index=False)
        print("\nLes résultats détaillés de la validation ont été sauvegardés dans 'resultats_validation_detailles.csv'")

    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc()