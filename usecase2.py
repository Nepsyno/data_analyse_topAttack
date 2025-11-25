# USE_CASE_FOOTBALL_LOGISTIC_REGRESSION_FINISSEURS.py (Corrigé)
# Modèle simple pour prédire les "Top Finisseurs"
# Dépendances: pandas, numpy, matplotlib, seaborn, scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi  # Import pour le radar
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# 1. --- CHARGEMENT & NETTOYAGE DES TYPES ---
csv_path = "dataset/2022-2023-football-player-stats.csv"
try:
    df = pd.read_csv(csv_path, sep=';', decimal=',', index_col='Rk', encoding='latin1')
except Exception as e:
    print(f"Erreur lors du chargement du CSV: {e}")
    exit()

# Conversion en numérique de TOUTES les colonnes potentiellement utiles
cols_to_numeric = [
    '90s', 'GCA', 'SCA', 'PasProg', 'CarProg',
    'Goals', 'Shots', 'SoT', 'SoT%', 'G/Sh', 'G/SoT',
    'ShoDist', 'ShoFK', 'ShoPK'
]
print("--- Conversion des types ---")
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Attention : Colonne '{col}' non trouvée.")

# 2. --- CRÉATION DE LA CIBLE "Top Finisseur" ---
print("\n--- Création de la cible (Top Finisseur) ---")

# Remplacer les valeurs infinies (ex: 1 But / 0 Tir Cadré = inf) par NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Filtres pour une analyse pertinente :
df_filtered = df[
    (df['90s'] > 1.0) &  # Au moins 1 match joué
    (df['Shots'] > 5)  # Au moins 6 tirs pour juger la finition
    ].copy()
print(f"Shape après filtrage (90s > 1.0 ET Shots > 5): {df_filtered.shape}")

# Imputer les NaN (ex: 0 Tirs Cadrés -> G/SoT = NaN) par la médiane AVANT de définir la cible
df_filtered['G/SoT'] = df_filtered['G/SoT'].fillna(df_filtered['G/SoT'].median())
df_filtered['SoT%'] = df_filtered['SoT%'].fillna(df_filtered['SoT%'].median())

# !! NOUVEAU !! : Calculer SCA_p90 pour le graphique
df_filtered['SCA_p90'] = df_filtered['SCA'] / df_filtered['90s']
df_filtered['SCA_p90'] = df_filtered['SCA_p90'].fillna(df_filtered['SCA_p90'].median())

# Règle de la cible: Au-dessus de la médiane en Efficacité (G/SoT) ET en Précision (SoT%)
gsot_median = df_filtered['G/SoT'].median()
sot_percent_median = df_filtered['SoT%'].median()
TARGET = 'TopAttacker'  # On garde ce nom, mais il signifie "Top Finisseur"

print(f"Médiane G/SoT (efficacité): {gsot_median:.2f}")
print(f"Médiane SoT% (précision): {sot_percent_median:.2f}")

df_filtered[TARGET] = np.where(
    (df_filtered['G/SoT'] > gsot_median) &
    (df_filtered['SoT%'] > sot_percent_median),
    1, 0
)
print(f"Distribution Cible '{TARGET}':\n", df_filtered[TARGET].value_counts())

# 3. --- PRETRAITEMENT (X et y) ---
# Utiliser le dataframe filtré et nettoyé
df_numeric = df_filtered.select_dtypes(include=[np.number])

# Gestion des Manquants restants (imputation par la médiane)
na_count = df_numeric.isna().sum().sum()
if na_count > 0:
    print(f"Imputation de {na_count} valeurs manquantes par la médiane...")
    df_numeric = df_numeric.fillna(df_numeric.median())

# 3.a Séparation X et y
y = df_numeric[TARGET]

# 3.b Définition des "HELPER_COLS" (Anti-Triche)
# On retire la cible ET TOUTES les colonnes liées à la finition
HELPER_COLS = [
    TARGET,
    'Goals', 'Shots', 'SoT', 'SoT%', 'G/Sh', 'G/SoT',
    'ShoDist', 'ShoFK', 'ShoPK', 'PKatt'
]
# On garde les colonnes "créatives" comme GCA/SCA/SCA_p90 pour voir si elles aident
# (elles seront aussi supprimées de X si elles sont dans HELPER_COLS, mais on les veut dans df_numeric)
# On s'assure de ne pas supprimer SCA_p90 de X, car c'est un prédicteur !
X = df_numeric.drop(columns=HELPER_COLS, errors='ignore')
print(f"\nColonnes Features (X) (extrait): {X.columns.tolist()[:10]}")
print(f"Colonnes exclues de X (car liées à la cible): {HELPER_COLS}")

# 3.c Mise à l'échelle (StandardScaler)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 3.d Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# 4. --- MODELISATION (Régression Logistique) ---
print("\n--- Entraînement du modèle (Régression Logistique) ---")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("Modèle entraîné.")

# 5. --- ÉVALUATION & ANALYSE (pour Partie 4) ---
print("\n--- Évaluation & Analyse (pour Partie 4) ---")
y_pred = model.predict(X_test)

print("\nClassification report (Performances du modèle):")
print(classification_report(y_test, y_pred, zero_division=0))

best_cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", best_cm)

# 5.a Variables les plus importantes (selon le modèle)
print(f"\n--- Variables les plus importantes pour prédire un 'Top Finisseur' ---")
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': coefficients})
feature_importance = feature_importance.reindex(
    feature_importance['Importance'].abs().sort_values(ascending=False).index)
print(feature_importance.head(15))  # Affiche les 15 plus importantes

# 6. --- VISUALISATIONS PERTINENTES (pour Recruteur) ---

# 6.a Graphique 1: Le "Cerveau" du Modèle (Feature Importance)
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15).sort_values(by='Importance', ascending=True)
colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
plt.xlabel('Impact sur la prédiction "Top Finisseur"')
plt.title('Quelles stats (non-liées aux tirs) prédisent la finition ?')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('logistic_finisseur_feature_importance_barchart.png')
print("\nSaved: logistic_finisseur_feature_importance_barchart.png")

# 6.b Graphique 2: La Carte des Talents (Quadrant Plot)
# (Utilise df_filtered qui contient maintenant G/SoT et SCA_p90)
plt.figure(figsize=(10, 8))

plot_data = df_filtered.copy()
# Ce "if" devrait maintenant fonctionner
if 'SCA_p90' in plot_data.columns and 'G/SoT' in plot_data.columns:

    plot_data['G/SoT_jitter'] = plot_data['G/SoT'] + np.random.uniform(-0.01, 0.01, size=len(plot_data))
    plot_data['SCA_p90_jitter'] = plot_data['SCA_p90'] + np.random.uniform(-0.01, 0.01, size=len(plot_data))

    sns.scatterplot(
        data=plot_data,
        x='SCA_p90_jitter',
        y='G/SoT_jitter',
        hue=TARGET,  # Coloré par notre CIBLE
        palette={0: 'gray', 1: 'green'},
        alpha=0.6,
        s=50
    )

    # Récupérer les médianes calculées à l'étape 2
    sca_p90_median = df_filtered['SCA_p90'].median()

    plt.axhline(gsot_median, color='red', linestyle='--', label=f'Médiane Finition (G/SoT = {gsot_median:.2f})')
    plt.axvline(sca_p90_median, color='blue', linestyle='--',
                label=f'Médiane Création (SCA_p90 = {sca_p90_median:.2f})')

    plt.title('Carte des Talents: Création vs Finition')
    plt.xlabel('Création (SCA_p90)')
    plt.ylabel('Finition (G/SoT)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig('logistic_finisseur_quadrant_plot.png')
    print("Saved: logistic_finisseur_quadrant_plot.png")
else:
    # Ce message ne devrait plus s'afficher
    print("Erreur: 'SCA_p90' ou 'G/SoT' non trouvé pour le quadrant plot.")

# 6.c Graphique 3: L'ADN du Finisseur (Profil Radar)
# Préparer les données pour le radar
radar_features = feature_importance.head(6)['Feature'].tolist()

# Combiner X_scaled et y pour faire la moyenne par groupe
data_for_radar = X_scaled.join(y)
radar_profiles = data_for_radar.groupby(TARGET)[radar_features].mean().reset_index()

# ---- Code du Radar Chart ----
categories = radar_features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Fermer le cercle

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for i, row in radar_profiles.iterrows():
    data = row[categories].tolist()
    data += data[:1]  # Fermer le cercle
    label = 'Non-Finisseur (Moyenne)' if row[TARGET] == 0 else 'Top Finisseur (Moyenne)'
    color = 'red' if row[TARGET] == 0 else 'green'

    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=label)
    ax.fill(angles, data, color=color, alpha=0.25)

plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.title('Profil Radar: "Top Finisseur" vs "Non-Finisseur"', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('logistic_finisseur_radar_profile.png')
print("Saved: logistic_finisseur_radar_profile.png")

# 6.d Matrice de confusion (On la garde pour l'évaluation)
plt.figure(figsize=(5, 4))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Usage Technique)')
plt.tight_layout()
plt.savefig('logistic_finisseur_confusion_matrix.png')
print("Saved: logistic_finisseur_confusion_matrix.png")

# 7. --- Sauvegarder résultats ---
feature_importance.to_csv('logistic_finisseur_feature_importance.csv', index=False)
print("Saved results CSVs: logistic_finisseur_feature_importance.csv")

# FIN