import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_RESULTS_CSV = 'resultats_validation_detailles.csv'
OUTPUT_REPORT_DIR = 'rapport_validation'

# Utiliser un style professionnel pour les graphiques
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['font.family'] = 'sans-serif' # Utilise une police standard
plt.rcParams['figure.dpi'] = 100 # Bonne résolution pour les images

# =============================================================================
# FONCTION PRINCIPALE DE REPORTING
# =============================================================================

def generate_validation_report(results_path: str):
    """
    Charge les résultats de la validation et génère un rapport complet
    avec des tableaux et des graphiques.
    """
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier de résultats '{results_path}' est introuvable.")
        print("Veuillez d'abord exécuter le script de validation.")
        return
        
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    print(f"Rapport généré dans le dossier : '{OUTPUT_REPORT_DIR}'")

    # --- 1. Générer le Tableau de Synthèse ---
    print("\n" + "="*50)
    print("      TABLEAU DE SYNTHÈSE DES PERFORMANCES")
    print("="*50)
    
    summary_data = []
    test_types = {
        'rank_perfect': 'Données Parfaites',
        'rank_partial': 'Données Partielles (50%)',
        'rank_noisy': 'Données Bruitées (2 erreurs)'
    }
    
    for col, name in test_types.items():
        top1 = (df[col] == 1).mean() * 100
        top5 = (df[col] <= 5).mean() * 100
        top10 = (df[col] <= 10).mean() * 100
        summary_data.append({
            'Scénario de Test': name,
            'Top-1 (%)': f"{top1:.2f}",
            'Top-5 (%)': f"{top5:.2f}",
            'Top-10 (%)': f"{top10:.2f}"
        })
        
    summary_df = pd.DataFrame(summary_data).set_index('Scénario de Test')
    print(summary_df.to_markdown()) # Affiche une jolie table formatée pour copier-coller

    # --- 2. Générer le Graphique en Barres Comparatif ---
    
    # Préparer les données pour le graphique
    plot_data = []
    for col, name in test_types.items():
        top1 = (df[col] == 1).mean()
        top5 = (df[col] <= 5).mean()
        top10 = (df[col] <= 10).mean()
        plot_data.extend([
            {'Scénario': name, 'Métrique': 'Top-1', 'Taux de Réussite': top1},
            {'Scénario': name, 'Métrique': 'Top-5', 'Taux de Réussite': top5},
            {'Scénario': name, 'Métrique': 'Top-10', 'Taux de Réussite': top10}
        ])
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x='Métrique', y='Taux de Réussite', hue='Scénario',
                     palette='viridis', edgecolor='black')

    # Ajouter les pourcentages sur les barres
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontweight='bold')

    plt.title("Performance du Système d'Identification par Scénario", fontsize=16, pad=20)
    plt.ylabel("Taux de Réussite", fontsize=12)
    plt.xlabel("Métrique d'Évaluation", fontsize=12)
    plt.ylim(0, 1.05) # Laisse de l'espace pour les annotations
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{i:.0%}" for i in np.arange(0, 1.1, 0.1)])
    plt.legend(title='Scénario de Test', loc='lower right')
    
    # Sauvegarder le graphique
    bar_chart_path = os.path.join(OUTPUT_REPORT_DIR, 'performance_comparative.png')
    plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraphique en barres sauvegardé dans : '{bar_chart_path}'")

    # --- 3. Générer l'Histogramme de Distribution des Rangs (pour le cas bruité) ---
    
    plt.figure(figsize=(12, 6))
    
    # Limiter l'affichage aux 50 premiers rangs pour une meilleure lisibilité
    ranks_to_plot = df['rank_noisy'][df['rank_noisy'] <= 50]
    
    ax = sns.histplot(ranks_to_plot, bins=50, kde=False, color=sns.color_palette("viridis")[3])
    
    plt.title("Distribution des Rangs d'Identification (Scénario Bruité)", fontsize=16, pad=20)
    plt.xlabel("Rang Obtenu", fontsize=12)
    plt.ylabel("Nombre de Taxons", fontsize=12)
    plt.yscale('log') # L'échelle logarithmique est idéale pour voir les petits nombres
    
    # Ajouter des lignes verticales pour Top-1, Top-5, Top-10
    plt.axvline(1, color='red', linestyle='--', label='Top-1')
    plt.axvline(5, color='blue', linestyle='--', label='Top-5')
    plt.axvline(10, color='orange', linestyle='--', label='Top-10')
    plt.legend()
    
    # Sauvegarder le graphique
    hist_path = os.path.join(OUTPUT_REPORT_DIR, 'distribution_rangs_bruites.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nHistogramme des rangs sauvegardé dans : '{hist_path}'")
    
    # --- 4. (Optionnel) Lister les "pires" identifications ---
    print("\n--- Analyse des identifications les moins performantes (Scénario Bruité) ---")
    worst_cases = df.sort_values('rank_noisy', ascending=False).head(10)
    print("Les 10 taxons les plus difficiles à identifier avec du bruit :")
    print(worst_cases[['taxon', 'rank_noisy']].to_string(index=False))


# =============================================================================
# EXÉCUTION DU SCRIPT DE REPORTING
# =============================================================================
if __name__ == "__main__":
    generate_validation_report(INPUT_RESULTS_CSV)