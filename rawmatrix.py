import pandas as pd
import sqlite3

# Nom du fichier de la base de données et du fichier de sortie
DB_FILE = 'mydb.db'  # Assurez-vous que le nom est correct, ex: 'mydb.db'
OUTPUT_CSV_FILE = 'matrice_brute_valeurs_reelles.csv'

def create_raw_matrix_with_real_values(db_path: str) -> pd.DataFrame:
    """
    Construit une matrice brute avec les valeurs d'observation réelles ('present',
    'absent', numériques, etc.) et une première ligne de poids experts.
    Cette version corrige l'erreur de "duplicate labels".
    """
    print(f"Connexion à la base de données '{db_path}'...")
    con = sqlite3.connect(db_path)

    # --- 1. Extraire les observations brutes des taxons ---
    print("Extraction des observations brutes des taxons...")
    sql_observations = """
    SELECT
        t.scientific_name,
        CASE
            WHEN n.node_type_id = 3 THEN parent_nl.label || ': ' || nl.label
            ELSE nl.label
        END AS descriptor_name,
        o.observed_value
    FROM observations o
    JOIN taxa t ON o.taxon_id = t.id AND t.rank_id = 25
    JOIN nodes n ON o.descriptor_node_id = n.id
    JOIN node_labels nl ON n.id = nl.node_id AND nl.language_code = 'fr'
    LEFT JOIN nodes_parents np ON n.id = np.child_id
    LEFT JOIN node_labels parent_nl ON np.parent_id = parent_nl.node_id AND parent_nl.language_code = 'fr'
    -- Assurer que le parent n'est pas nul pour les descripteurs d'état
    WHERE (n.node_type_id != 3 OR parent_nl.label IS NOT NULL);
    """
    df_obs = pd.read_sql_query(sql_observations, con)
    
    # Supprimer les éventuels doublons d'observations (un taxon ne devrait avoir qu'une valeur par descripteur)
    df_obs.drop_duplicates(subset=['scientific_name', 'descriptor_name'], keep='first', inplace=True)

    matrix_taxa = pd.pivot_table(
        df_obs,
        index='scientific_name',
        columns='descriptor_name',
        values='observed_value',
        aggfunc='first'
    )

    # --- 2. Extraire les poids experts ---
    print("Extraction des poids experts...")
    sql_weights = """
    SELECT DISTINCT -- Ajout de DISTINCT pour aider à dédoublonner à la source
        CASE
            WHEN n.node_type_id = 3 THEN parent_nl.label || ': ' || nl.label
            ELSE nl.label
        END AS descriptor_name,
        n.discriminant_power_on20 AS weight
    FROM nodes n
    JOIN node_labels nl ON n.id = nl.node_id AND nl.language_code = 'fr'
    LEFT JOIN nodes_parents np ON n.id = np.child_id
    LEFT JOIN node_labels parent_nl ON np.parent_id = parent_nl.node_id AND parent_nl.language_code = 'fr'
    WHERE n.discriminant_power_on20 IS NOT NULL
      AND (n.node_type_id != 3 OR parent_nl.label IS NOT NULL);
    """
    df_weights = pd.read_sql_query(sql_weights, con)

    # *** CORRECTION PRINCIPALE ICI ***
    # Si des doublons de 'descriptor_name' persistent, on les agrège.
    # On prend la moyenne des poids, ce qui est une stratégie sûre si les valeurs sont identiques.
    print(f"Nombre de poids extraits avant dédoublonnage : {len(df_weights)}")
    df_weights_unique = df_weights.groupby('descriptor_name').agg({'weight': 'mean'}).reset_index()
    print(f"Nombre de poids uniques après dédoublonnage : {len(df_weights_unique)}")
    
    if len(df_weights_unique) < len(df_weights):
        print("Avertissement : Des noms de descripteurs dupliqués ont été trouvés et agrégés.")

    s_weights = pd.Series(df_weights_unique.weight.values, index=df_weights_unique.descriptor_name)
    s_weights.name = 'weight'

    # --- 3. Combiner pour former la matrice finale ---
    print("Combinaison des données pour former la matrice finale...")
    
    s_weights_aligned = s_weights.reindex(matrix_taxa.columns)
    df_weights_aligned = s_weights_aligned.to_frame().T
    
    final_matrix = pd.concat([df_weights_aligned, matrix_taxa])
    
    mean_weight = s_weights_aligned.mean()
    if pd.isna(mean_weight): mean_weight = 0 # Cas où il n'y a aucun poids
    
    final_matrix.loc['weight'] = final_matrix.loc['weight'].fillna(mean_weight)
    final_matrix.fillna('unknown', inplace=True)
    
    con.close()
    
    return final_matrix

# --- Script principal ---
if __name__ == "__main__":
    try:
        # Remplacez 'botany.db' par 'mydb.db' si c'est le nom de votre fichier
        matrice_brute = create_raw_matrix_with_real_values(DB_FILE)

        matrice_brute.reset_index(inplace=True)
        matrice_brute.rename(columns={'index': 'ID'}, inplace=True)
        
        matrice_brute.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\nOpération terminée avec succès !")
        print(f"La matrice brute a été sauvegardée dans : '{OUTPUT_CSV_FILE}'")
        print("\nAperçu de la matrice finale :")
        # display(matrice_brute.head(6))
        print(matrice_brute.head(6))

    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")
        import traceback
        traceback.print_exc() # Imprime plus de détails sur l'erreur
        print("Veuillez vous assurer que le fichier de base de données existe et est accessible.")