import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DISTANCE_FOLDER = 'resultats_distance'
INPUT_MATRIX_FILE = 'matrice_prete_pour_calcul.csv'
ALPHA_VALUES_TO_TEST = [0.0, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

def analyze_separability():
    """
    Analyse la capacité de chaque alpha à séparer les espèces
    du même genre par rapport à celles de genres différents.
    """
    try:
        df_full = pd.read_csv(INPUT_MATRIX_FILE)
        taxa_names = df_full[df_full['ID'] != 'weight']['ID'].values
        # Extraire le genre de chaque nom d'espèce
        genres = [name.split(' ')[0] for name in taxa_names]
        df_taxa_info = pd.DataFrame({'taxon': taxa_names, 'genre': genres})
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{INPUT_MATRIX_FILE}' est introuvable.")
        return

    if not os.path.exists(DISTANCE_FOLDER):
        print(f"Erreur : Le dossier '{DISTANCE_FOLDER}' est introuvable.")
        return

    # Stocker les résultats de l'analyse
    separability_scores = []

    for filename in sorted(os.listdir(DISTANCE_FOLDER)):
        if not filename.endswith(".csv"):
            continue
        
        try:
            alpha = float(filename.split('_')[-1].replace('.csv', ''))
            if alpha not in ALPHA_VALUES_TO_TEST: continue
        except (IndexError, ValueError):
            continue

        print(f"\n--- Analyse pour alpha = {alpha:.2f} ---")
        filepath = os.path.join(DISTANCE_FOLDER, filename)
        df_distance = pd.read_csv(filepath, index_col=0)
        
        # S'assurer que la matrice est alignée avec nos infos de taxons
        df_distance = df_distance.reindex(index=taxa_names, columns=taxa_names)
        
        distances_intra_genre = []
        distances_inter_genre = []
        
        # Itérer sur toutes les paires uniques de taxons
        for i in range(len(taxa_names)):
            for j in range(i + 1, len(taxa_names)):
                taxon1 = taxa_names[i]
                taxon2 = taxa_names[j]
                genre1 = genres[i]
                genre2 = genres[j]
                
                distance = df_distance.loc[taxon1, taxon2]
                
                if genre1 == genre2:
                    distances_intra_genre.append(distance)
                else:
                    distances_inter_genre.append(distance)

        # Calculer les statistiques
        mean_intra = np.mean(distances_intra_genre) if distances_intra_genre else 0
        mean_inter = np.mean(distances_inter_genre) if distances_inter_genre else 0
        
        # Le score de séparabilité est la différence entre les moyennes
        separability_score = mean_inter - mean_intra
        
        separability_scores.append({'alpha': alpha, 
                                    'separability': separability_score,
                                    'mean_intra_genre_dist': mean_intra,
                                    'mean_inter_genre_dist': mean_inter})

        print(f"  Distance moyenne intra-genre: {mean_intra:.4f}")
        print(f"  Distance moyenne inter-genre: {mean_inter:.4f}")
        print(f"  Score de séparabilité: {separability_score:.4f}")
        
    # --- Visualisation ---
    if not separability_scores:
        print("Aucun résultat à visualiser.")
        return
        
    results_df = pd.DataFrame(separability_scores).sort_values('alpha')

    # Graphique principal : Score de séparabilité en fonction de alpha
    optimal_alpha_df = results_df.loc[results_df['separability'].idxmax()]
    optimal_alpha = optimal_alpha_df['alpha']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    sns.lineplot(data=results_df, x='alpha', y='separability', marker='o', ax=ax1, color='red', label='Score de Séparabilité (Inter - Intra)')
    ax1.set_xlabel("Valeur de Alpha", fontsize=12)
    ax1.set_ylabel("Score de Séparabilité", color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    
    # Ajouter un deuxième axe y pour les distances moyennes
    ax2 = ax1.twinx()
    sns.lineplot(data=results_df, x='alpha', y='mean_intra_genre_dist', marker='s', ax=ax2, color='blue', linestyle='--', label='Dist. moyenne Intra-Genre')
    sns.lineplot(data=results_df, x='alpha', y='mean_inter_genre_dist', marker='^', ax=ax2, color='green', linestyle='--', label='Dist. moyenne Inter-Genre')
    ax2.set_ylabel("Distance Moyenne", fontsize=12)
    
    plt.axvline(x=optimal_alpha, color='black', linestyle=':', label=f'Alpha Optimal ({optimal_alpha:.2f})')
    
    fig.suptitle("Analyse de Séparabilité en fonction d'Alpha", fontsize=16)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.xticks(ALPHA_VALUES_TO_TEST, rotation=45)
    plt.show()

if __name__ == "__main__":
    analyze_separability()