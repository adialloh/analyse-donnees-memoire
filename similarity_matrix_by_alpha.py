import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import os

# Noms des fichiers et constantes
INPUT_MATRIX_FILE = 'matrice_prete_pour_calcul.csv'
OUTPUT_FOLDER = 'resultats_similarite' # Dossier pour stocker les fichiers CSV

# Votre liste de valeurs alpha à tester
ALPHA_VALUES_TO_TEST = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
EPSILON = 1e-9

class RobustTanimotoIdentifier:
    """
    Classe pour calculer la similarité de Tanimoto robuste et pondérée.
    (Code identique à la réponse précédente)
    """
    def __init__(self, full_matrix: pd.DataFrame):
        if 'ID' not in full_matrix.columns:
            raise ValueError("La matrice doit contenir une colonne 'ID'.")
        matrix = full_matrix.set_index('ID')
        self.expert_weights = matrix.loc['weight']
        self.taxa_matrix = matrix.drop('weight')
        print(f"Initialisation avec {len(self.taxa_matrix)} taxons et {len(self.expert_weights)} descripteurs.")
        self._calculate_frequency_weights()

    def _calculate_frequency_weights(self):
        N = len(self.taxa_matrix)
        n_e = self.taxa_matrix.sum(axis=0)
        self.frequency_weights = np.log(N / (n_e + 1))
        self.frequency_weights.replace([np.inf, -np.inf], 0, inplace=True)

    def get_hybrid_weights(self, alpha: float):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha doit être compris entre 0 et 1.")
        P_expert = self.expert_weights
        P_frequence = self.frequency_weights
        return (alpha * P_expert) + ((1 - alpha) * P_frequence)

    def calculate_similarity_matrix(self, alpha: float, epsilon: float = 1e-9):
        print(f"\nCalcul de la matrice de similarité pour alpha = {alpha:.2f}...")
        
        w = self.get_hybrid_weights(alpha).values
        
        # Fonction de distance pour scikit-learn
        def distance_function(u, v):
            dot = np.sum(w * u * v)
            norm_u = np.sum(w * u**2)
            norm_v = np.sum(w * v**2)
            denom = norm_u + norm_v - dot
            similarity = np.clip(dot / (denom + epsilon), 0, 1)
            return 1.0 - similarity

        distance_matrix = pairwise_distances(self.taxa_matrix.values, metric=distance_function, n_jobs=-1) # Utilise tous les coeurs CPU
        similarity_matrix = 1 - distance_matrix
        
        df_similarity = pd.DataFrame(similarity_matrix, 
                                     index=self.taxa_matrix.index, 
                                     columns=self.taxa_matrix.index)
        print("Calcul terminé.")
        return df_similarity

# --- Script principal ---
def generate_all_similarity_matrices():
    """
    Script principal qui charge les données et génère un fichier de similarité CSV pour chaque alpha.
    """
    try:
        # Charger la matrice numérique propre
        df_full_matrix = pd.read_csv(INPUT_MATRIX_FILE)
        
        # Créer le dossier de sortie s'il n'existe pas
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"Dossier '{OUTPUT_FOLDER}' créé pour stocker les résultats.")
            
        # Initialiser l'identificateur
        identifier = RobustTanimotoIdentifier(df_full_matrix)
        
        # Boucle sur toutes les valeurs d'alpha à tester
        for alpha in ALPHA_VALUES_TO_TEST:
            
            # Calculer la matrice de similarité pour l'alpha courant
            similarity_matrix = identifier.calculate_similarity_matrix(alpha=alpha, epsilon=EPSILON)
            
            # Définir un nom de fichier clair
            # Le format :.2f garantit que l'alpha est écrit avec deux décimales (ex: 0.80)
            output_filename = f"similarite_alpha_{alpha:.2f}.csv"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Sauvegarder la matrice dans son fichier CSV
            similarity_matrix.to_csv(output_path, encoding='utf-8-sig')
            
            print(f"-> Matrice de similarité sauvegardée dans : '{output_path}'")
        
        print("\n--- Opération terminée ---")
        print(f"Tous les fichiers de similarité ont été générés dans le dossier '{OUTPUT_FOLDER}'.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{INPUT_MATRIX_FILE}' n'a pas été trouvé.")
        print("Veuillez exécuter les scripts précédents pour le générer.")
    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_all_similarity_matrices()