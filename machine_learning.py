import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _prepare_and_filter_data(df, min_minutes=450):
    """
    Prepare numeric data and filter players by minimum minutes.
    Returns filtered dataframe and boolean mask.
    """
    df_num = df.drop(columns=["Rk", "Player", "Nation", "Pos", "Squad", "Comp"], errors='ignore')
    df_num = df_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    df_num['Minutes'] = df_num.get('90s', 0) * 90
    mask = df_num['Minutes'] >= min_minutes

    if mask.sum() == 0:
        raise ValueError("No players pass the minutes filter. Lower min_minutes.")

    return df_num[mask].copy(), mask


def _engineer_features(num):
    """
    Create all engineered features from raw stats.
    Modifies dataframe in place.
    """
    # Regularized per-90 metrics
    denominator = num['90s'] + 2

    per90_features = {
        'Goals_per90_reg': 'Goals',
        'Shots_per90_reg': 'Shots',
        'SoT_per90_reg': 'SoT',
        'SCA_per90': 'SCA',
        'GCA_per90': 'GCA',
        'PasAss_per90': 'PasAss',
        'TklAtt3rd_per90': 'TklAtt3rd',
        'TouAtt3rd_per90': 'TouAtt3rd'
    }

    for new_col, raw_col in per90_features.items():
        num[new_col] = num[raw_col] / denominator

    # Stabilized ratio features
    num['G_Sh_adj'] = np.log1p((num.get('G/Sh', 0).replace(0, 0) * num.get('Shots', 0)).clip(lower=0))
    num['G_SoT_adj'] = np.log1p((num.get('G/SoT', 0).replace(0, 0) * num.get('SoT', 0)).clip(lower=0))


def _get_feature_columns():
    """Return the list of stable features used for scoring."""
    return [
        'Goals_per90_reg', 'G_Sh_adj', 'G_SoT_adj',
        'Shots_per90_reg', 'SoT_per90_reg',
        'SCA_per90', 'GCA_per90', 'PasAss_per90',
        'TklAtt3rd_per90', 'TouAtt3rd_per90'
    ]


def _get_feature_weights():
    """Return the weights for each feature in AttackScore calculation."""
    return {
        'Goals_per90_reg': 0.35,
        'G_Sh_adj': 0.18,
        'G_SoT_adj': 0.17,
        'Shots_per90_reg': 0.08,
        'SoT_per90_reg': 0.07,
        'SCA_per90': 0.08,
        'GCA_per90': 0.05,
        'PasAss_per90': 0.02
    }


def compute_attack_score(df, min_minutes=450, top_quantile=0.90):
    """
    Compute stable, production-ready AttackScore:
    - Filters low-minute players
    - Regularizes per-90 rates
    - Stabilizes ratio features with log1p
    - Removes unstable features
    - Scales stable features and computes weighted AttackScore
    """
    num, mask = _prepare_and_filter_data(df, min_minutes)
    _engineer_features(num)

    cols = _get_feature_columns()
    weights = _get_feature_weights()

    # Scale features
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(num[cols]), columns=cols, index=num.index)

    # Compute weighted AttackScore
    attack = sum(scaled[feature] * weight for feature, weight in weights.items())

    # Attach results back to full df
    df['AttackScore'] = np.nan
    df.loc[num.index, 'AttackScore'] = attack.values

    # Determine top attackers
    threshold = attack.quantile(top_quantile)
    df['TopAttacker'] = 0
    df.loc[attack.index, 'TopAttacker'] = (attack >= threshold).astype(int).values

    return df


def analyze_feature_importance(df, min_minutes=450):
    """
    Analyze feature importance using correlations with AttackScore.
    Visualizes which features are most important for predicting top attackers.
    """
    num, _ = _prepare_and_filter_data(df, min_minutes)
    _engineer_features(num)

    feature_cols = _get_feature_columns()

    # Calculate correlation with AttackScore
    num_with_score = num[feature_cols].copy()
    num_with_score['AttackScore'] = df.loc[num.index, 'AttackScore']

    correlations = num_with_score.corr()['AttackScore'].drop('AttackScore')
    feature_importance = pd.DataFrame({
        'Feature': correlations.index,
        'Importance': np.abs(correlations.values)
    }).sort_values('Importance', ascending=True)

    # Print analysis
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print("\nCorrelation with AttackScore (absolute values):")
    print(feature_importance.sort_values('Importance', ascending=False).to_string(index=False))

    # Generate visualizations
    _plot_feature_distribution(feature_importance)
    _plot_ranked_features(feature_importance)
    _plot_top_features_scatter(num_with_score, feature_importance)

    return feature_importance


def _plot_feature_distribution(feature_importance):
    """Plot pie chart of feature contribution percentage."""
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.nlargest(7, 'Importance')
    other_importance = feature_importance[~feature_importance['Feature'].isin(top_features['Feature'])][
        'Importance'].sum()

    pie_data = list(top_features['Importance']) + [other_importance]
    pie_labels = list(top_features['Feature']) + ['Others']
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))

    plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('Feature Importance Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _plot_ranked_features(feature_importance):
    """Plot all features ranked by importance."""
    plt.figure(figsize=(12, 6))
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=False)
    colors_rank = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_sorted)))

    plt.barh(range(len(feature_importance_sorted)),
             feature_importance_sorted['Importance'].values,
             color=colors_rank)
    plt.yticks(range(len(feature_importance_sorted)),
               feature_importance_sorted['Feature'].values)
    plt.xlabel('Correlation Strength', fontsize=12, fontweight='bold')
    plt.title('All Features Ranked by Importance', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(feature_importance_sorted.iterrows()):
        plt.text(row['Importance'], i, f" {row['Importance']:.4f}",
                 va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def _plot_top_features_scatter(num_with_score, feature_importance):
    """Plot scatter plot of top 3 features vs AttackScore."""
    plt.figure(figsize=(10, 7))
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=False)
    top_3_features = feature_importance_sorted.head(3)['Feature'].values
    colors_scatter = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, feature in enumerate(top_3_features):
        plt.scatter(num_with_score[feature], num_with_score['AttackScore'],
                    alpha=0.6, s=80, label=feature, color=colors_scatter[i])

    plt.xlabel('Feature Value', fontsize=12, fontweight='bold')
    plt.ylabel('AttackScore', fontsize=12, fontweight='bold')
    plt.title('Top 3 Features vs AttackScore', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('dataset/2022-2023-football-player-stats.csv', sep=';', encoding="latin1")

    # Compute stable AttackScore
    df = compute_attack_score(df, min_minutes=450, top_quantile=0.90)

    # Filter valid scores for analysis
    df_valid = df[df['AttackScore'].notna()].copy()

    # Display top attackers
    top_players = df_valid[df_valid["TopAttacker"] == 1].sort_values("AttackScore", ascending=False)
    print(
        f"Number of Top Attackers: {len(top_players)} out of {len(df_valid)} valid players (filtered >= 450 minutes).")
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

    # Analyze Feature Importance
    feature_importance = analyze_feature_importance(df, min_minutes=450)