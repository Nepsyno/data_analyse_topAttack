import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# FEATURE ENGINEERING & SCORING
# ============================================================================

def compute_attack_score(df, min_minutes=450, top_quantile=0.90):
    """
    Compute AttackScore for all players based on offensive statistics.
    Players below min_minutes are excluded.
    """
    # Clean and prepare data
    df_num = df.drop(columns=["Rk", "Player", "Nation", "Pos", "Squad", "Comp"], errors='ignore')
    df_num = df_num.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_num['Minutes'] = df_num.get('90s', 0) * 90

    # Filter by minimum minutes
    mask = df_num['Minutes'] >= min_minutes
    num = df_num[mask].copy()

    if mask.sum() == 0:
        raise ValueError("No players meet minimum minutes requirement")

    # Create engineered features
    denom = num['90s'] + 2  # Regularization

    num['Goals_per90'] = num['Goals'] / denom
    num['Shots_per90'] = num['Shots'] / denom
    num['SoT_per90'] = num['SoT'] / denom
    num['SCA_per90'] = num['SCA'] / denom
    num['GCA_per90'] = num['GCA'] / denom
    num['G_Sh_adj'] = np.log1p((num.get('G/Sh', 0) * num.get('Shots', 0)).clip(lower=0))
    num['G_SoT_adj'] = np.log1p((num.get('G/SoT', 0) * num.get('SoT', 0)).clip(lower=0))

    # Select features for scoring
    features = ['Goals_per90', 'G_Sh_adj', 'G_SoT_adj', 'Shots_per90', 'SoT_per90', 'SCA_per90', 'GCA_per90']

    # Normalize features
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(num[features]), columns=features, index=num.index)

    # Weighted score
    weights = {
        'Goals_per90': 0.35,
        'G_Sh_adj': 0.25,
        'G_SoT_adj': 0.20,
        'Shots_per90': 0.10,
        'SoT_per90': 0.05,
        'SCA_per90': 0.03,
        'GCA_per90': 0.02
    }

    attack_score = sum(scaled[feat] * weights[feat] for feat in features)

    # Attach scores to original dataframe
    df['AttackScore'] = np.nan
    df.loc[num.index, 'AttackScore'] = attack_score.values

    # Label top attackers (90th percentile)
    threshold = attack_score.quantile(top_quantile)
    df['TopAttacker'] = 0
    df.loc[attack_score.index, 'TopAttacker'] = (attack_score >= threshold).astype(int).values

    return df, num, features


# ============================================================================
# VISUALIZATION - ATTACK SCORE ANALYSIS
# ============================================================================

def plot_attack_score_distribution(df_valid):
    """Display distribution of AttackScore and identify top attackers."""
    plt.figure(figsize=(12, 6))

    threshold = df_valid["AttackScore"].quantile(0.90)

    # Background scatter
    plt.scatter(range(len(df_valid)), df_valid["AttackScore"].values,
                alpha=0.5, s=60, c='#2E86AB', label='Players')

    # Highlight top attackers
    top_mask = df_valid["AttackScore"] >= threshold
    plt.scatter(np.where(top_mask)[0], df_valid.loc[top_mask, "AttackScore"].values,
                alpha=0.8, s=150, c='#FF6B6B', marker='*', label='Top Attackers', edgecolors='darkred', linewidth=1)

    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90th Percentile Threshold')

    plt.title('Attack Score Distribution - Top 10% Identified', fontsize=14, fontweight='bold')
    plt.xlabel('Player Index', fontsize=12, fontweight='bold')
    plt.ylabel('Attack Score', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(df, num, features):
    """Analyze and visualize feature importance via correlation with AttackScore."""
    num_with_score = num[features].copy()
    num_with_score['AttackScore'] = df.loc[num.index, 'AttackScore']

    corr = num_with_score.corr()['AttackScore'].drop('AttackScore')
    importance_df = pd.DataFrame({
        'Feature': corr.index,
        'Importance': np.abs(corr.values)
    }).sort_values('Importance', ascending=True)

    # Print
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Correlation with AttackScore)")
    print("=" * 70)
    print(importance_df.sort_values('Importance', ascending=False).to_string(index=False))

    # Visualize
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df)))

    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    plt.xlabel('Correlation Strength', fontsize=12, fontweight='bold')
    plt.title('Feature Importance for Attack Score', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(importance_df['Importance']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return importance_df


# ============================================================================
# KNN MODEL - HYPERPARAMETER TUNING
# ============================================================================

def train_knn_with_tuning(df, num, features, min_minutes=450):
    """Train KNN with manual hyperparameter search and visualization."""

    X = num[features].copy()
    y = df.loc[num.index, 'TopAttacker'].astype(int)

    # Parameter ranges - use more reasonable test size range
    k_values = np.arange(3, 20, 1)
    test_sizes = np.arange(0.2, 0.5, 0.05)

    print("\n" + "=" * 70)
    print("KNN HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Testing {len(k_values)} k values and {len(test_sizes)} test_size values")

    # Grid search with cross-validation
    results_acc = np.zeros((len(test_sizes), len(k_values)))
    results_f1 = np.zeros((len(test_sizes), len(k_values)))
    results_cv = np.zeros((len(test_sizes), len(k_values)))  # Cross-validation scores

    for i, test_size in enumerate(test_sizes):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for j, k in enumerate(k_values):
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

            # Cross-validation on training set
            cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1')
            results_cv[i, j] = cv_scores.mean()

            # Test set performance
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)

            results_acc[i, j] = accuracy_score(y_test, y_pred)
            results_f1[i, j] = f1_score(y_test, y_pred)

    # Find best parameters based on cross-validation (more reliable than test set)
    best_idx_cv = np.unravel_index(results_cv.argmax(), results_cv.shape)
    best_k = k_values[best_idx_cv[1]]
    best_test_size = test_sizes[best_idx_cv[0]]

    print(f"\nBest k: {best_k}")
    print(f"Best test_size: {best_test_size:.2f}")
    print(f"Best CV F1 Score: {results_cv[best_idx_cv]:.4f}")
    print(f"Test F1 Score: {results_f1[best_idx_cv]:.4f}")

    # Check for overfitting
    overfit_diff = results_f1[best_idx_cv] - results_cv[best_idx_cv]
    if overfit_diff > 0.1:
        print(f"⚠️  WARNING: Possible overfitting detected! Test score - CV score = {overfit_diff:.4f}")

    # Visualize heatmap using CV scores (more reliable)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(results_cv, annot=True, fmt='.3f', cmap='YlOrRd', cbar=True,
                xticklabels=k_values, yticklabels=[f'{ts:.2f}' for ts in test_sizes],
                ax=ax, cbar_kws={'label': 'CV F1 Score'})

    # Highlight best
    rect = plt.Rectangle((best_idx_cv[1], best_idx_cv[0]), 1, 1, fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)

    plt.title(
        f'KNN Hyperparameter Tuning - Cross-Validation F1 Score\n(Best: k={best_k}, test_size={best_test_size:.2f})',
        fontsize=14, fontweight='bold')
    plt.xlabel('Number of Neighbors (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Size Ratio', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return best_k, best_test_size, X, y


# ============================================================================
# FINAL KNN MODEL & EVALUATION
# ============================================================================

def train_final_knn_model(X, y, best_k, best_test_size):
    """Train final KNN model and evaluate performance."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=best_test_size, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = knn.predict(X_test_scaled)

    print("\n" + "=" * 70)
    print("FINAL KNN MODEL - TEST SET PERFORMANCE")
    print("=" * 70)
    print(f"Train set: {len(X_train)} players | Test set: {len(X_test)} players")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Top Attacker', 'Top Attacker']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Top', 'Top'], yticklabels=['Not Top', 'Top'])
    plt.title('Confusion Matrix - KNN Classifier', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return knn, X_train_scaled, X_test_scaled, scaler


# ============================================================================
# PCA VISUALIZATION - CLUSTERS & ATTACK SCORE
# ============================================================================

def visualize_knn_with_pca(df, X, knn, scaler, num, features):
    """Visualize KNN clusters using PCA with 2 different colorings."""

    # Scale all data and get predictions
    X_scaled = scaler.fit_transform(X)
    predictions = knn.predict(X_scaled)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("\n" + "=" * 70)
    print("PCA VISUALIZATION")
    print("=" * 70)

    # AttackScore Gradient Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    mask_not_top = predictions == 0
    mask_top = predictions == 1

    attack_scores = df.loc[num.index, 'AttackScore'].values

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=attack_scores, cmap='YlOrRd', alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
    ax.scatter(X_pca[mask_top, 0], X_pca[mask_top, 1],
                c='darkred', alpha=0.9, s=250, marker='*', edgecolors='darkred', linewidth=2, label='Top Attacker',
                zorder=10)

    ax.set_xlabel(f'PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2', fontsize=12, fontweight='bold')
    ax.set_title('Players - Colored by AttackScore', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='AttackScore')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == '__main__':
    # Load and compute scores
    print("Loading data and computing AttackScore...")
    df = pd.read_csv('dataset/2022-2023-football-player-stats.csv', sep=';', encoding="latin1")
    df, num, features = compute_attack_score(df, min_minutes=450, top_quantile=0.90)

    # Display results
    df_valid = df[df['AttackScore'].notna()].copy()
    top_players = df_valid[df_valid["TopAttacker"] == 1].sort_values("AttackScore", ascending=False)

    print(f"\n{'=' * 70}")
    print(f"TOP ATTACKERS: {len(top_players)} out of {len(df_valid)} players (>450 minutes)")
    print(f"{'=' * 70}")
    print(top_players[["Player", "Squad", "Goals", "AttackScore"]].head(20).to_string(index=False))

    # Visualizations
    plot_attack_score_distribution(df_valid)
    importance_df = plot_feature_importance(df, num, features)

    # KNN tuning and training
    best_k, best_test_size, X, y = train_knn_with_tuning(df, num, features)
    knn, X_train_scaled, X_test_scaled, scaler = train_final_knn_model(X, y, best_k, best_test_size)

    # Final visualization
    visualize_knn_with_pca(df, X, knn, scaler, num, features)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)