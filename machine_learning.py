import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_attack_score(df, min_minutes=450, top_quantile=0.90):
    """
    Compute stable, production-ready AttackScore:
    - Filters low-minute players
    - Regularizes per-90 rates
    - Stabilizes ratio features with log1p
    - Removes unstable features
    - Scales stable features and computes weighted AttackScore
    """
    # prepare numeric copy
    df_num = df.drop(columns=["Rk", "Player", "Nation", "Pos", "Squad", "Comp"], errors='ignore')
    df_num = df_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    # filter players with too few minutes
    df_num['Minutes'] = df_num.get('90s', 0) * 90
    mask = df_num['Minutes'] >= min_minutes
    if mask.sum() == 0:
        raise ValueError("No players pass the minutes filter. Lower min_minutes.")

    df_f = df[mask].copy()
    num = df_num[mask].copy()

    # regularized per90 (add pseudo-count to denominator to stabilize small samples)
    num['Goals_per90_reg'] = num['Goals'] / (num['90s'] + 2)
    num['Shots_per90_reg'] = num['Shots'] / (num['90s'] + 2)
    num['SoT_per90_reg'] = num['SoT'] / (num['90s'] + 2)
    num['SCA_per90'] = num['SCA'] / (num['90s'] + 2)
    num['GCA_per90'] = num['GCA'] / (num['90s'] + 2)
    num['PasAss_per90'] = num['PasAss'] / (num['90s'] + 2)
    num['TklAtt3rd_per90'] = num['TklAtt3rd'] / (num['90s'] + 2)
    num['TouAtt3rd_per90'] = num['TouAtt3rd'] / (num['90s'] + 2)

    # stabilize ratio features using volume-aware log1p
    num['G_Sh_adj'] = np.log1p((num.get('G/Sh', 0).replace(0, 0) * num.get('Shots', 0)).clip(lower=0))
    num['G_SoT_adj'] = np.log1p((num.get('G/SoT', 0).replace(0, 0) * num.get('SoT', 0)).clip(lower=0))

    # choose stable features (drop very unstable ones like raw CPA)
    cols = [
        'Goals_per90_reg', 'G_Sh_adj', 'G_SoT_adj',
        'Shots_per90_reg', 'SoT_per90_reg',
        'SCA_per90', 'GCA_per90', 'PasAss_per90',
        'TklAtt3rd_per90', 'TouAtt3rd_per90'
    ]

    # scale and compute weighted AttackScore (weights sum ~= 1)
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(num[cols]), columns=cols, index=num.index)

    attack = (
        scaled['Goals_per90_reg'] * 0.35 +
        scaled['G_Sh_adj'] * 0.18 +
        scaled['G_SoT_adj'] * 0.17 +
        scaled['Shots_per90_reg'] * 0.08 +
        scaled['SoT_per90_reg'] * 0.07 +
        scaled['SCA_per90'] * 0.08 +
        scaled['GCA_per90'] * 0.05 +
        scaled['PasAss_per90'] * 0.02
    )

    # attach results back to full df
    df['AttackScore'] = np.nan
    df.loc[num.index, 'AttackScore'] = attack.values

    # determine top attackers among filtered players only
    threshold = attack.quantile(top_quantile)
    df['TopAttacker'] = 0
    df.loc[attack.index, 'TopAttacker'] = (attack >= threshold).astype(int).values

    return df


# Load data
df = pd.read_csv('dataset/2022-2023-football-player-stats.csv', sep=';', encoding="latin1")

# Compute stable AttackScore
df = compute_attack_score(df, min_minutes=450, top_quantile=0.90)

# Filter valid scores for analysis
df_valid = df[df['AttackScore'].notna()].copy()

# Display top attackers
top_players = df_valid[df_valid["TopAttacker"] == 1].sort_values("AttackScore", ascending=False)
print(f"Number of Top Attackers: {len(top_players)} out of {len(df_valid)} valid players (filtered >= 450 minutes).")
print(top_players[["Player", "Squad", "Goals", "AttackScore"]].to_string(index=False))

# Visualize distribution
plt.figure(figsize=(10, 6))
threshold = df_valid["AttackScore"].quantile(0.90)
plt.scatter(range(len(df_valid)), df_valid["AttackScore"].values, alpha=0.6)
plt.axhline(y=threshold, color='r', linestyle='--', label='Top Attacker Threshold (90th percentile)')
plt.title('Distribution of Attack Scores (Stable Pro Scoring)')
plt.xlabel('Player Index (filtered >= 450 minutes)')
plt.ylabel('Attack Score')
plt.legend()
plt.show()
