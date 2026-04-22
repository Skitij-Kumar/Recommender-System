import pandas as pd
import numpy as np


def compute_group_preference(df, weights):
    return (df * weights[:, None]).sum(axis=0) / weights.sum()


def compute_user_align(df, prev_group_pref):
    if prev_group_pref is None:
        return np.ones(df.shape[0])

    align = 1 - np.abs(df.values - prev_group_pref.values).mean(axis=1)
    return align


def WIAA(df, iterations=10, lambda_=0.5, rehab_factor=0.01):
    n_users = df.shape[0]
    weights = np.ones(n_users) #for equal influence
    prev_group_pref = None 

    for _ in range(iterations):
        group_pref = compute_group_preference(df, weights)
        divergence = 1 - \
            (1 - np.abs(df.values - group_pref.values).mean(axis=1))
        user_align = compute_user_align(df, prev_group_pref)

        weights = (1 - lambda_) * (1 - divergence) + lambda_ * user_align
        weights = weights * (1 - rehab_factor) + rehab_factor
        prev_group_pref = group_pref.copy()

    final_group_pref = compute_group_preference(df, weights)
    return final_group_pref, weights


def main():
    df = pd.read_csv('ratings.csv')

    group_input = input("Enter users for group (e.g. 1 2 3): ")
    group = [int(u) for u in group_input.strip().split()]
    group_df = df[df['userId'].isin(group)]

    # Users as rows, movies as columns
    # safe pivot, normalize to [0,1], then fill NaNs
    ratings_matrix = group_df.pivot_table(
        index='userId', columns='movieId', values='rating', aggfunc='mean'
    )
    ratings_matrix = (ratings_matrix - 1.0) / 4.0   # MovieLens 1-5 -> [0,1]
    ratings_matrix = ratings_matrix.fillna(0)

    final_group_pref, final_weights = WIAA(
        ratings_matrix, iterations=10, lambda_=0.5, rehab_factor=0.01)

    # Sort preferences
    sorted_prefs = final_group_pref.sort_values(ascending=False)
 
    print("\nGroup member weights:")
    for user, weight in zip(ratings_matrix.index, final_weights):
        print(f"User {user}: {weight:.3f}")

    print("\nTop 5 most recommended items:")
    print(sorted_prefs.head())

    print("\nBottom 5 least recommended items:")
    print(sorted_prefs.tail())


if __name__ == "__main__":
    main()
