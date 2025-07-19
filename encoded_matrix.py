import pandas as pd
import numpy as np

# Noms des fichiers d'entrée et de sortie
INPUT_CSV_FILE = 'matrice_brute_valeurs_reelles.csv'
OUTPUT_CSV_FILE = 'matrice_encodee.csv'

def encode_matrix(input_path: str, output_path: str):
    """
    Charge une matrice brute, encode les valeurs textuelles en valeurs numériques
    selon des règles spécifiées, et sauvegarde le résultat.
    """
    print(f"Chargement de la matrice brute depuis '{input_path}'...")
    try:
        # Charger la matrice. La première colonne ('ID') sera utilisée comme index.
        df = pd.read_csv(input_path, index_col='ID')
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{input_path}' n'a pas été trouvé.")
        print("Veuillez vous assurer d'avoir d'abord exécuté le script de création de la matrice brute.")
        return

    print("Encodage des valeurs en cours...")
    
    # Créer une copie pour ne pas modifier le DataFrame original pendant l'itération
    df_encoded = df.copy()

    # Définir le dictionnaire de remplacement
    # C'est une méthode très efficace et lisible pour remplacer plusieurs valeurs à la fois.
    replacement_map = {
        'present': 1.0,         # Convertir en float pour la cohérence avec NaN
        'absent': 0.0,
        'inapplicable': 0.0,
        'unknown': np.nan       # np.nan est la représentation standard de NaN/NA dans pandas
    }

    # Appliquer le remplacement à toutes les colonnes, sauf la ligne 'weight'
    # On sélectionne toutes les lignes SAUF 'weight'
    rows_to_encode = df_encoded.index != 'weight'
    
    # La méthode .replace() est parfaite pour cela
    df_encoded.loc[rows_to_encode] = df_encoded.loc[rows_to_encode].replace(replacement_map)

    # Pour les colonnes qui pourraient contenir des valeurs numériques sous forme de texte,
    # on les convertit en vrais nombres. On ignore les erreurs pour les colonnes qui restent textuelles.
    # On applique cela à toutes les colonnes, y compris la ligne de poids.
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='ignore')

    print("Encodage terminé.")
    
    # Réinsérer la colonne 'ID' à partir de l'index pour la sauvegarde
    df_encoded.reset_index(inplace=True)
    df_encoded.rename(columns={'index': 'ID'}, inplace=True)
    
    # Sauvegarder la nouvelle matrice encodée
    df_encoded.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nOpération terminée avec succès !")
    print(f"La matrice encodée a été sauvegardée dans : '{output_path}'")
    print("\nAperçu de la matrice encodée :")
    print(df_encoded.head(6))


# --- Script principal ---
if __name__ == "__main__":
    encode_matrix(INPUT_CSV_FILE, OUTPUT_CSV_FILE)