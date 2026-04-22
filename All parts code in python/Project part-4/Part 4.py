import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# top_k = number of items to include in the final group list


def greedy_grow(user_item_matrix, group_user_ids, top_k=10):
    # items rated by group users
    remaining_items = set(user_item_matrix.columns)
    group_list = []

    # Each user's individual preferences (dict)
    user_preferences = {uid: user_item_matrix.loc[uid].sort_values(ascending=False).to_dict()
                        for uid in group_user_ids}

    for _ in range(top_k):
        item_scores = {}

        # avg.score for each item among group members
        for item in remaining_items:
            score = np.mean([user_preferences[uid].get(item, 0)
                            for uid in group_user_ids])
            item_scores[item] = score

        if not item_scores:
            break

        best_item = max(item_scores, key=item_scores.get)
        group_list.append(best_item)  # add item with best satisfaction to list

        # remove item from already existing lists
        for uid in group_user_ids:
            user_preferences[uid].pop(best_item, None)
        remaining_items.remove(best_item)

    return group_list


def predict_for_group(item_id, group_user_ids, user_sim_matrix, user_item_matrix):
    ratings = []
    for uid in group_user_ids:
        sims = user_sim_matrix.loc[uid, group_user_ids].drop(
            uid)  # similarity to other users
        weighted_sum = 0
        sim_sum = 0

        # Weighted sum of neighbors' ratings
        for neighbor_uid, sim in sims.items():
            rating = user_item_matrix.at[neighbor_uid,
                                         item_id] if item_id in user_item_matrix.columns else 0
            weighted_sum += rating * sim
            sim_sum += sim
        ratings.append(weighted_sum / sim_sum if sim_sum > 0 else 0)
    return np.mean(ratings)


def find_counterfactual_minimal(target_item, user_item_matrix, group_user_ids, user_sim_matrix, topN=10, max_combo=2, candidate_items=None):
    # pool = only items any group user interacted with, excluding target
    pool = [c for c in user_item_matrix.columns if c != target_item and (
        user_item_matrix.loc[group_user_ids, c] > 0).sum() > 1]
    if candidate_items is None:
        candidate_items = pool

    def removes_target(items_to_remove):
        temp = user_item_matrix.copy()
        for it in items_to_remove:
            temp.loc[group_user_ids, it] = 0

        # predicted scores after removal
        scores = [(itm, predict_for_group(itm, group_user_ids, user_sim_matrix, temp))
                  for itm in candidate_items]

        # sorting and getting top-N items
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = [itm for itm, _ in scores[:topN]]

        # Checking if target item is removed from top-N
        return target_item not in top_items

    # Removing single items first
    for it in pool:
        if (user_item_matrix.loc[group_user_ids, it] > 0).sum() <= 1:
            continue
        if removes_target([it]):
            return [it]

    # Trying combinations of items
    for size in range(2, max_combo + 1):
        for combo in combinations(pool, size):
            users_count = sum(
                any(user_item_matrix.loc[uid, c] > 0 for c in combo) for uid in group_user_ids)
            if users_count <= 1:
                continue
            if removes_target(combo):
                return list(combo)

    return None


def main():
    # prepare data
    data = pd.read_csv("ratings.csv")
    user_item_matrix = data.pivot(
        index="userId", columns="movieId", values="rating").fillna(0)

    # input group users
    group_input = input("Enter users for group (e.g., 1 2 3): ")
    group_user_ids = [int(u) for u in group_input.strip().split()]

    # Restrict matrices to selected group
    group_user_item_matrix = user_item_matrix.loc[group_user_ids]
    group_user_sim_matrix = pd.DataFrame(
        cosine_similarity(group_user_item_matrix),
        index=group_user_ids,
        columns=group_user_ids
    )

    # group recommendation using greedy (top_k=10)
    greedy = greedy_grow(group_user_item_matrix, group_user_ids, top_k=10)
    final_list = greedy

    # scoring and ranking
    movie_scores = [(mid, predict_for_group(mid, group_user_ids, group_user_sim_matrix, group_user_item_matrix))
                    for mid in final_list]
    movie_scores.sort(key=lambda x: x[1], reverse=True)

    print("Recommendations and counterfactual explanations:")
    for mid, score in movie_scores[:5]:
        explanation = find_counterfactual_minimal(
            mid, group_user_item_matrix, group_user_ids, group_user_sim_matrix, topN=10, max_combo=2, candidate_items=final_list)
        print(f"{mid}, {score}, explanation: {explanation}")


if __name__ == "__main__":
    main()
