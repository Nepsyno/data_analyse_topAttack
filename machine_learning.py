import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1) LOAD DATA
# =============================================================================
print("Loading dataset...")
df = pd.read_csv("dataset/2022-2023-football-player-stats.csv", sep=";", encoding="latin1")
print("Dataset loaded:", df.shape)

# =============================================================================
# 2) CLEAN + FILTER MINUTES
# =============================================================================
min_minutes = 450
top_quantile = 0.90

# Keep only numeric columns for computation (ignore non-numeric identifiers)
df_num = df.drop(columns=["Rk", "Player", "Nation", "Pos", "Squad", "Comp"], errors="ignore")
df_num = df_num.apply(pd.to_numeric, errors="coerce").fillna(0)

# Minutes from 90s
df_num["Minutes"] = df_num.get("90s", 0) * 90

mask = df_num["Minutes"] >= min_minutes
num = df_num[mask].copy()

if mask.sum() == 0:
    raise ValueError("No players meet minimum minutes requirement")

print(f"Players kept (>= {min_minutes} minutes):", len(num))

# =============================================================================
# 3) FEATURE ENGINEERING
# =============================================================================
denom = num["90s"] + 2  # regularization

num["Goals_per90"] = num["Goals"] / denom
num["Shots_per90"] = num["Shots"] / denom
num["SoT_per90"] = num["SoT"] / denom
num["SCA_per90"] = num["SCA"] / denom
num["GCA_per90"] = num["GCA"] / denom

num["G_Sh_adj"] = np.log1p((num.get("G/Sh", 0) * num.get("Shots", 0)).clip(lower=0))
num["G_SoT_adj"] = np.log1p((num.get("G/SoT", 0) * num.get("SoT", 0)).clip(lower=0))

features = [
    "Goals_per90",
    "G_Sh_adj",
    "G_SoT_adj",
    "Shots_per90",
    "SoT_per90",
    "SCA_per90",
    "GCA_per90",
]

# =============================================================================
# 4) NORMALIZATION + ATTACK SCORE
# =============================================================================
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(num[features]), columns=features, index=num.index)

weights = {
    "Goals_per90": 0.35,
    "G_Sh_adj": 0.25,
    "G_SoT_adj": 0.20,
    "Shots_per90": 0.10,
    "SoT_per90": 0.05,
    "SCA_per90": 0.03,
    "GCA_per90": 0.02,
}

attack_score = (
    scaled["Goals_per90"] * weights["Goals_per90"]
    + scaled["G_Sh_adj"] * weights["G_Sh_adj"]
    + scaled["G_SoT_adj"] * weights["G_SoT_adj"]
    + scaled["Shots_per90"] * weights["Shots_per90"]
    + scaled["SoT_per90"] * weights["SoT_per90"]
    + scaled["SCA_per90"] * weights["SCA_per90"]
    + scaled["GCA_per90"] * weights["GCA_per90"]
)

df["AttackScore"] = np.nan
df.loc[num.index, "AttackScore"] = attack_score.values

# =============================================================================
# 5) TOP ATTACKER LABEL
# =============================================================================
threshold = attack_score.quantile(top_quantile)

df["TopAttacker"] = 0
df.loc[attack_score.index, "TopAttacker"] = (attack_score >= threshold).astype(int).values

df_valid = df[df["AttackScore"].notna()].copy()
top_players = df_valid[df_valid["TopAttacker"] == 1].sort_values("AttackScore", ascending=False)

print("\n" + "=" * 70)
print(f"TOP ATTACKERS: {len(top_players)} out of {len(df_valid)} players (>= {min_minutes} minutes)")
print("=" * 70)
print(top_players[["Player", "Squad", "Goals", "AttackScore"]].head(20).to_string(index=False))

# =============================================================================
# 6) VISUALIZATION: ATTACK SCORE DISTRIBUTION
# =============================================================================
print("\nPlotting AttackScore distribution...")
plt.figure(figsize=(12, 6))

plt.scatter(
    range(len(df_valid)),
    df_valid["AttackScore"].values,
    alpha=0.5,
    s=60,
    label="Players",
)

top_mask = df_valid["AttackScore"] >= threshold
plt.scatter(
    np.where(top_mask)[0],
    df_valid.loc[top_mask, "AttackScore"].values,
    alpha=0.8,
    s=150,
    marker="*",
    label="Top Attackers",
    edgecolors="black",
    linewidth=0.8,
)

plt.axhline(y=threshold, linestyle="--", linewidth=2, label="90th Percentile Threshold")

plt.title("Attack Score Distribution - Top 10% Identified")
plt.xlabel("Player Index")
plt.ylabel("Attack Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# 7) FEATURE IMPORTANCE (CORRELATION WITH ATTACK SCORE)
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Correlation with AttackScore)")
print("=" * 70)

num_with_score = num[features].copy()
num_with_score["AttackScore"] = df.loc[num.index, "AttackScore"]

corr = num_with_score.corr(numeric_only=True)["AttackScore"].drop("AttackScore")
importance_df = pd.DataFrame({"Feature": corr.index, "Importance": np.abs(corr.values)}).sort_values(
    "Importance", ascending=False
)

print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
plt.xlabel("Correlation Strength")
plt.title("Feature Importance for Attack Score")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
