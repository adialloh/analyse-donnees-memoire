import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# Noms des fichiers
INPUT_MATRIX_FILE = 'matrice_prete_pour_calcul.csv'
# Le fichier de sortie sera une matrice de similarité pour un alpha spécifique
OUTPUT_SIMILARITY_FILE = 'matrice_similarite_alpha_0.8.csv'

class RobustTanimotoIdentifier:
    """
    Calcule la similarité de Tanimoto robuste et pondérée.
    S = sum(w*A*B) / (sum(w*A^2) + sum(w*B^2) - sum(w*A*B) + epsilon)
    """
    def __init__(self, full_matrix: pd.DataFrame):
        """
        Initialise l'identificateur avec la matrice complète contenant les poids.
        
        Args:
            full_matrix (pd.DataFrame): DataFrame chargé depuis 'matrice_prete_pour_calcul.csv'.
                                        La première colonne doit être 'ID'.
        """
        if 'ID' not in full_matrix.columns:
            raise ValueError("La matrice doit contenir une colonne 'ID'.")

        # Étape 1: Séparer les poids et les données des taxons
        matrix = full_matrix.set_index('ID')
        self.expert_weights = matrix.loc['weight']
        self.taxa_matrix = matrix.drop('weight')
        
        print(f"Initialisation avec {len(self.taxa_matrix)} taxons et {len(self.expert_weights)} descripteurs.")
        
        # Étape 2: Calculer les poids de fréquence
        self._calculate_frequency_weights()

    def _calculate_frequency_weights(self):
        """
        Calcule les poids basés sur la fréquence (rareté) des descripteurs.
        poids_frequence = log(N / (n_e + 1))
        """
        N = len(self.taxa_matrix)  # Nombre total de taxons
        # Pour les données normalisées (0-1), sum() est une approximation de n_e.
        # Pour les données binaires, c'est exactement n_e.
        n_e = self.taxa_matrix.sum(axis=0)
        
        self.frequency_weights = np.log(N / (n_e + 1))
        
        # Remplacer les infinis si n_e=0 (le log peut devenir infini)
        self.frequency_weights.replace([np.inf, -np.inf], 0, inplace=True)
        print("Poids de fréquence (log(N/(n_e+1))) calculés.")

    def get_hybrid_weights(self, alpha: float):
        """
        Calcule le poids final hybride 'w_i' pour un alpha donné.
        w_i = alpha * poids_expert + (1 - alpha) * poids_frequence
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha doit être compris entre 0 et 1.")
            
        P_expert = self.expert_weights
        P_frequence = self.frequency_weights
        
        hybrid_weights = (alpha * P_expert) + ((1 - alpha) * P_frequence)
        return hybrid_weights.values

    def get_similarity_function(self, alpha: float, epsilon: float = 1e-9):
        """
        Retourne une fonction de SIMILARITÉ Tanimoto pondérée pour un alpha spécifique.
        """
        w = self.get_hybrid_weights(alpha)
        
        def weighted_tanimoto_similarity(u: np.ndarray, v: np.ndarray) -> float:
            """
            S_w(A, B) = sum(w*A*B) / (sum(w*A^2) + sum(w*B^2) - sum(w*A*B) + epsilon)
            """
            weighted_dot_product = np.sum(w * u * v)
            weighted_norm_u_sq = np.sum(w * u**2)
            weighted_norm_v_sq = np.sum(w * v**2)
            
            denominator = weighted_norm_u_sq + weighted_norm_v_sq - weighted_dot_product
            
            # Formule robuste avec epsilon
            similarity = weighted_dot_product / (denominator + epsilon)
            
            # S'assurer que la similarité est entre 0 et 1
            return np.clip(similarity, 0, 1)
            
        return weighted_tanimoto_similarity

    def calculate_similarity_matrix(self, alpha: float, epsilon: float = 1e-9):
        """
        Calcule la matrice de similarité complète pour un alpha donné.
        NOTE: sklearn.pairwise_distances calcule des distances, donc nous calculons 1 - similarité.
        """
        print(f"\nCalcul de la matrice de similarité pour alpha = {alpha:.2f}...")
        
        # Pour utiliser pairwise_distances, nous devons lui donner une fonction de DISTANCE
        w = self.get_hybrid_weights(alpha)
        
        def distance_function(u, v):
            sim = weighted_tanimoto_similarity(u, v, w, epsilon)
            return 1.0 - sim

        # Fonction interne pour le calcul de similarité, utilisée par la fonction de distance
        def weighted_tanimoto_similarity(u, v, w_local, eps_local):
            dot = np.sum(w_local * u * v)
            norm_u = np.sum(w_local * u**2)
            norm_v = np.sum(w_local * v**2)
            denom = norm_u + norm_v - dot
            return np.clip(dot / (denom + eps_local), 0, 1)

        # Calculer la matrice de distance
        distance_matrix = pairwise_distances(self.taxa_matrix.values, metric=distance_function)
        
        # La matrice de similarité est 1 - la matrice de distance
        similarity_matrix = 1 - distance_matrix
        
        # Convertir en DataFrame pour une meilleure lisibilité
        df_similarity = pd.DataFrame(similarity_matrix, 
                                     index=self.taxa_matrix.index, 
                                     columns=self.taxa_matrix.index)
        
        print("Calcul terminé.")
        return df_similarity


# --- Script principal ---
if __name__ == "__main__":
    try:
        # Charger la matrice numérique propre
        df_full_matrix = pd.read_csv(INPUT_MATRIX_FILE)
        
        # Initialiser l'identificateur
        identifier = RobustTanimotoIdentifier(df_full_matrix)
        
        # Choisir une valeur d'alpha pour le calcul
        alpha_to_use = 0.8
        
        # Calculer la matrice de similarité
        similarity_matrix = identifier.calculate_similarity_matrix(alpha=alpha_to_use)
        
        # Sauvegarder la matrice de similarité dans un fichier CSV
        similarity_matrix.to_csv(OUTPUT_SIMILARITY_FILE, encoding='utf-8-sig')
        
        print(f"\nOpération terminée avec succès !")
        print(f"La matrice de similarité pour alpha={alpha_to_use} a été sauvegardée dans : '{OUTPUT_SIMILARITY_FILE}'")
        
        print("\nAperçu de la matrice de similarité finale :")
        print(similarity_matrix.iloc[:6, :6])

        # --- Exemple d'utilisation de la matrice de similarité ---
        print("\n--- Exemple : Trouver les 5 taxons les plus similaires à 'Abrus precatorius' ---")
        target_taxon = 'Abrus precatorius'
        if target_taxon in similarity_matrix.index:
            # Sélectionner la ligne du taxon cible, enlever sa propre similarité (qui est 1)
            # et trier les autres par ordre décroissant de similarité.
            top_5_similar = similarity_matrix[target_taxon].drop(target_taxon).sort_values(ascending=False).head(5)
            print(top_5_similar)
        else:
            print(f"Le taxon '{target_taxon}' n'a pas été trouvé dans la matrice.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{INPUT_MATRIX_FILE}' n'a pas été trouvé.")
        print("Veuillez exécuter les scripts précédents pour le générer.")
    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc()