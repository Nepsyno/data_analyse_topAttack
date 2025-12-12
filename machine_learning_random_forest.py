import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(importance_df)))

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
# RANDOM FOREST WITH RANDOMIZEDSEARCHCV
# ============================================================================

def train_random_forest_with_tuning(df, num, features):
    """Train Random Forest with RandomizedSearchCV for hyperparameter optimization."""

    X = num[features].copy()
    y = df.loc[num.index, 'TopAttacker'].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST - HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")
    print("=" * 70)
    print(f"Train set: {len(X_train)} players | Test set: {len(X_test)} players")
    print(f"Top Attackers - Train: {y_train.sum()} | Test: {y_test.sum()}\n")

    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
    }

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # RandomizedSearchCV - teste 30 combinaisons aléatoires
    print("Running RandomizedSearchCV (30 iterations)...")
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=30,  # Nombre de combinaisons à tester
        cv=5,  # 5-fold cross-validation
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    random_search.fit(X_train_scaled, y_train)

    print("\n✓ RandomizedSearchCV Completed!")
    print(f"\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  - {param}: {value}")

    print(f"\nBest CV F1 Score: {random_search.best_score_:.4f}")

    # Get best model
    best_rf = random_search.best_estimator_

    # Evaluate on test set
    y_pred = best_rf.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Check overfitting
    overfit_diff = test_f1 - random_search.best_score_
    if overfit_diff > 0.1:
        print(f"⚠️  WARNING: Possible overfitting! Diff = {overfit_diff:.4f}")
    elif overfit_diff < -0.15:
        print(f"⚠️  WARNING: Possible underfitting! Diff = {overfit_diff:.4f}")
    else:
        print(f"✓ Good balance (Diff = {overfit_diff:.4f})")

    # Visualize hyperparameter search results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    top_10 = results_df.head(10)
    iterations = range(1, len(top_10) + 1)
    colors_iter = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10)))

    plt.barh(iterations, top_10['mean_test_score'].values, color=colors_iter)
    plt.xlabel('Mean CV F1 Score', fontsize=12, fontweight='bold')
    plt.ylabel('Top 10 Iterations', fontsize=12, fontweight='bold')
    plt.title('Top 10 Best Hyperparameter Combinations', fontsize=13, fontweight='bold')
    plt.xlim([0, 1])

    for i, v in enumerate(top_10['mean_test_score'].values):
        plt.text(v + 0.02, i + 1, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.subplot(1, 2, 2)
    iterations_all = range(1, len(results_df) + 1)
    plt.scatter(iterations_all, results_df['mean_test_score'].values,
                alpha=0.6, s=80, c=results_df['mean_test_score'].values, cmap='RdYlGn', edgecolors='black',
                linewidth=0.5)
    plt.axhline(y=random_search.best_score_, color='green', linestyle='--', linewidth=2,
                label=f'Best Score: {random_search.best_score_:.3f}')
    plt.xlabel('Iteration Number', fontsize=12, fontweight='bold')
    plt.ylabel('CV F1 Score', fontsize=12, fontweight='bold')
    plt.title('RandomizedSearchCV - All 30 Iterations', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return best_rf, scaler, X_train_scaled, X_test_scaled, X, y, y_test, y_pred


# ============================================================================
# RANDOM FOREST EVALUATION
# ============================================================================

def evaluate_random_forest(y_test, y_pred, best_rf, X_test_scaled, features):
    """Display detailed evaluation metrics and confusion matrix."""

    print("\n" + "=" * 70)
    print("RANDOM FOREST - TEST SET PERFORMANCE")
    print("=" * 70)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Top Attacker', 'Top Attacker']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Top', 'Top'], yticklabels=['Not Top', 'Top'])

    # Add percentages
    for i in range(2):
        for j in range(2):
            total = cm[i].sum()
            pct = cm[i, j] / total * 100 if total > 0 else 0
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                     ha='center', va='center', fontsize=10, color='gray', fontweight='bold')

    plt.title('Confusion Matrix - Random Forest Classifier', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Feature Importance from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 70)
    print(feature_importance.to_string(index=False))

    # Visualize Feature Importance
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_importance)))

    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Feature Importance - Random Forest Model', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(feature_importance['Importance'].values):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return feature_importance


# ============================================================================
# PCA VISUALIZATION
# ============================================================================

def visualize_rf_with_pca(df, X, best_rf, scaler, num):
    """Visualize Random Forest predictions using PCA."""

    # Scale all data and get predictions
    X_scaled = scaler.fit_transform(X)
    predictions = best_rf.predict(X_scaled)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("\n" + "=" * 70)
    print("PCA VISUALIZATION - RANDOM FOREST PREDICTIONS")
    print("=" * 70)
    print(f"PC1 explains {pca.explained_variance_ratio_[0] * 100:.1f}% of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1] * 100:.1f}% of variance")
    print(f"Total: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    mask_not_top = predictions == 0
    mask_top = predictions == 1

    attack_scores = df.loc[num.index, 'AttackScore'].values

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=attack_scores, cmap='YlOrRd', alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
    ax.scatter(X_pca[mask_top, 0], X_pca[mask_top, 1],
               c='darkred', alpha=0.9, s=250, marker='*', edgecolors='darkred', linewidth=2,
               label='Top Attacker (Predicted)',
               zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Predictions - Colored by AttackScore', fontsize=13, fontweight='bold')
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

    # Random Forest with RandomizedSearchCV
    best_rf, scaler, X_train_scaled, X_test_scaled, X, y, y_test, y_pred = train_random_forest_with_tuning(df, num,
                                                                                                           features)

    # Evaluation
    feature_importance = evaluate_random_forest(y_test, y_pred, best_rf, X_test_scaled, features)

    # Final visualization
    visualize_rf_with_pca(df, X, best_rf, scaler, num)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
