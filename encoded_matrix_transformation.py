import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# Noms des fichiers
INPUT_CSV_FILE = 'matrice_encodee.csv'
OUTPUT_CSV_FILE = 'matrice_prete_pour_calcul.csv'

def parse_numerical_value(value):
    """
    Analyse une chaîne de caractères pour en extraire une valeur numérique.
    Gère les intervalles, les approximations et les bornes.
    Retourne un float si possible, sinon np.nan.
    """
    if pd.isna(value):
        return np.nan

    # Convertir en chaîne de caractères pour le traitement
    s_value = str(value).strip().lower()

    # Si la valeur est déjà un nombre valide, la retourner
    try:
        return float(s_value)
    except ValueError:
        pass # Continuer le parsing si ce n'est pas un simple nombre

    # Étape 1: Nettoyer la chaîne (enlever unités, etc.)
    # Garde les chiffres, les points, les virgules, les tirets et les caractères < >
    s_value = re.sub(r'[a-z"\'\s]+', '', s_value)
    s_value = s_value.replace(',', '.') # Standardiser les décimales

    # Étape 2: Gérer les intervalles (ex: "3-5")
    if '-' in s_value:
        try:
            parts = [float(p) for p in s_value.split('-')]
            return np.mean(parts)
        except (ValueError, IndexError):
            return np.nan # L'intervalle est mal formé

    # Étape 3: Gérer les approximations et les bornes (ex: "<5", ">10")
    # On retire les caractères non-numériques restants et on essaie de convertir
    try:
        # re.sub() retire tout ce qui n'est pas un chiffre ou un point
        cleaned_num = re.sub(r'[^\d\.]', '', s_value)
        if cleaned_num:
            return float(cleaned_num)
    except ValueError:
        return np.nan

    return np.nan # Si aucune règle ne correspond

def process_advanced_data(input_path: str, output_path: str):
    """
    Charge la matrice brute, la nettoie en utilisant le parsing avancé,
    puis la transforme (imputation, normalisation).
    """
    print(f"Chargement de la matrice brute depuis '{input_path}'...")
    try:
        df_full = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{input_path}' n'a pas été trouvé.")
        return

    # --- Étape 1: Séparer les Poids et les Données ---
    df_weights = df_full[df_full['ID'] == 'weight'].set_index('ID')
    df_taxa_raw = df_full[df_full['ID'] != 'weight'].set_index('ID')
    print(f"{len(df_taxa_raw)} taxons et {len(df_weights.columns)} descripteurs chargés.")

    # --- Étape 2: Appliquer le parsing et l'encodage ---
    print("\nApplication du parsing avancé et de l'encodage...")
    
    # Créer une nouvelle matrice pour les données parsées
    df_taxa_parsed = df_taxa_raw.copy()

    # Dictionnaire d'encodage pour les valeurs catégorielles
    replacement_map = {
        'present': 1.0,
        'absent': 0.0,
        'inapplicable': 0.0,
        'unknown': np.nan
    }

    # Appliquer le remplacement et le parsing sur chaque cellule
    for col in df_taxa_parsed.columns:
        # La méthode .apply() permet d'appliquer une fonction à chaque élément d'une colonne
        df_taxa_parsed[col] = df_taxa_parsed[col].apply(
            lambda x: replacement_map.get(str(x).lower(), x) # Remplacer si c'est une clé connue
        ).apply(parse_numerical_value) # Parser le reste
    
    print("Parsing terminé.")

    # --- Étape 3: Identifier les colonnes et imputer les NaN ---
    print("\nÉtape d'imputation (remplacement des NaN)...")
    numerical_cols = []
    for col in df_taxa_parsed.columns:
        if df_taxa_parsed[col].dropna().nunique() > 2:
            numerical_cols.append(col)
    
    for col in df_taxa_parsed.columns:
        if col in numerical_cols:
            mean_val = df_taxa_parsed[col].mean()
            df_taxa_parsed[col].fillna(mean_val, inplace=True)
            print(f"  -> NaN dans la colonne numérique '{col}' remplacés par la moyenne ({mean_val:.2f}).")
        else: # Colonne binaire
            df_taxa_parsed[col].fillna(0, inplace=True)
            
    print("  -> NaN dans les colonnes binaires remplacés par 0.")
    
    # --- Étape 4: Mise à l'Échelle (Normalisation Min-Max) ---
    print("\nÉtape de normalisation (Min-Max Scaling)...")
    if numerical_cols:
        scaler = MinMaxScaler()
        df_taxa_parsed[numerical_cols] = scaler.fit_transform(df_taxa_parsed[numerical_cols])
        print(f"  -> {len(numerical_cols)} colonne(s) numérique(s) ont été normalisées sur une échelle de [0, 1].")
    else:
        print("  -> Aucune colonne numérique à normaliser.")
        
    # --- Étape 5: Recombiner et Sauvegarder ---
    print("\nRecombinaison de la matrice finale...")
    df_final = pd.concat([df_weights[df_taxa_parsed.columns], df_taxa_parsed])
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'ID'}, inplace=True)

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nOpération terminée avec succès !")
    print(f"La matrice prête pour le calcul a été sauvegardée dans : '{output_path}'")
    print("\nAperçu de la matrice finale transformée :")
    print(df_final.head(6))


# --- Script principal ---
if __name__ == "__main__":
    process_advanced_data(INPUT_CSV_FILE, OUTPUT_CSV_FILE)