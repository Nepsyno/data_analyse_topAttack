import pandas as pd

df = pd.read_csv('dataset/2022-2023-football-player-stats.csv', sep=';')
df_attaquants = df[
    (df['Pos'].str.contains('FW', na=False)) & 
    (df['Min'] >= 300)
].copy()

print(f"Dataset initial : {df.shape}")
print(f"Dataset filtré (Attaquants) : {df_attaquants.shape}")

cols_to_keep = [
    'Player', 'Squad', 'Age',       # Identité
    'Goals', 'Shots', 'SoT',        # Finition brute
    'SoT%', 'Assists',              # Précision et Altruisme
    'SCA', 'GCA',                   # Création (Shot/Goal Creating Actions)
    'TouAttPen',                    # Présence : Touches dans la surface de réparation
    'CarProg'                       # Percussion : Conduites de balle progressives
]

df_final = df_attaquants[cols_to_keep].copy()

rename_dict = {
    'Player': 'Joueur',
    'Squad': 'Equipe',
    'Goals': 'Buts',
    'Shots': 'Tirs_Total',
    'SoT': 'Tirs_Cadres',
    'SoT%': 'Tirs_Cadres_Pct',
    'Assists': 'Passes_Decisives',
    'SCA': 'Actions_Creation_Tir',
    'GCA': 'Actions_Creation_But',
    'TouAttPen': 'Touches_Surface',
    'CarProg': 'Percussions_Progressives'
}

df_final = df_final.rename(columns=rename_dict)
df_final = df_final.reset_index(drop=True)
display(df_final.head())

df_final.to_csv('top_attaquants_data.csv', index=False)