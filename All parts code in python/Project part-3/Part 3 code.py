import pandas as pd
import numpy as np


def cosine_similarity_matrix(X):
    # Computing cosine similarity between movies based on genres
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norm + 1e-10)
    return np.dot(X_normalized, X_normalized.T)


def compute_group_preference(df, weights):
    # Computing weighted average of user ratings for group preference
    w = np.array(weights).reshape(-1, 1)
    result = (df.values * w).sum(axis=0) / w.sum()
    return pd.Series(result, index=df.columns)


def compute_user_align(df, prev_group_pref):
    # Computing alignment of each user with previous group preference
    if prev_group_pref is None:
        return np.ones(len(df))
    return 1 - np.abs(df.values - prev_group_pref.values).mean(axis=1)


def WIAA(df, iterations=10, lambda_=0.5, rehab_factor=0.01):
    # Weighted Iterative Averaging Algorithm for group preference
    weights = np.ones(len(df))
    prev_group_pref = None

    for _ in range(iterations):
        group_pref = compute_group_preference(df, weights)
        divergence = np.abs(df - group_pref).mean(axis=1)
        align = compute_user_align(df, prev_group_pref)
        # Updating user weights for combining divergence and alignment
        weights = (1 - lambda_) * (1 - divergence) + lambda_ * align
        # Applying rehabilitation factor to avoid zero weights
        weights = weights * (1 - rehab_factor) + rehab_factor
        prev_group_pref = group_pref

    final_group_pref = compute_group_preference(df, weights)
    return final_group_pref, weights


def mmr(cand_scores, sim_matrix, history, k=5, lam=0.8, beta=0.35):
    # Selecting diverse items using MMR (Maximal Marginal Relevance)
    selected = []

    while len(selected) < k and not cand_scores.empty:
        best_item, best_val = None, -1e9
        for item, score in cand_scores.items():
            # Similarity with already selected items
            sim_sel = max([sim_matrix.loc[item, s] for s in selected] + [0])
            # Similarity with history
            sim_hist = max([sim_matrix.loc[item, h] for h in history] + [0])
            v = lam * score - (1 - lam) * sim_sel - beta * sim_hist
            if v > best_val:
                best_val, best_item = v, item

        if best_item is None:
            break

        selected.append(best_item)
        cand_scores = cand_scores.drop(best_item)

    return selected


def main():
    # Loading ratings and movies data
    df = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')

    # Getting user input for group
    group_input = input("Enter users for group (e.g. 1 2 3): ")
    group = [int(u) for u in group_input.strip().split()]
    group_df = df[df['userId'].isin(group)]

    # Creating user-item ratings matrix
    ratings_matrix = group_df.pivot_table(
        index='userId', columns='movieId', values='rating', aggfunc='mean')
    ratings_matrix = ((ratings_matrix - 1.0) / 4.0).fillna(0)

    # Computing final group preferences using WIAA
    final_group_pref, _ = WIAA(
        ratings_matrix, iterations=10, lambda_=0.5, rehab_factor=0.01)
    candidates = final_group_pref.sort_values(ascending=False).head(100)

    # Creating genre matrix for movies
    G = movies.set_index('movieId')['genres'].fillna(
        '').str.get_dummies(sep='|')
    sim_matrix = pd.DataFrame(
        cosine_similarity_matrix(G.values),
        index=G.index,
        columns=G.index
    )

    # Recommendation rounds
    history = []  # Track all recommended items
    ROUNDS, K = 3, 5  # Number of rounds and items per round
    for r in range(1, ROUNDS + 1):
        remaining = candidates.drop(history, errors='ignore')
        selected = mmr(remaining, sim_matrix, history, k=K)
        print(f"Round {r} selected: {selected}")
        history.extend(selected)


if __name__ == "__main__":
    main()
