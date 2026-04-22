import pandas as pd 
import numpy as np

def pearson(ratings):
    similarity = ratings.T.corr(method='pearson', min_periods=2)
    return similarity

def kendall(ratings):
    similarity = ratings.T.corr(method='kendall', min_periods=2)
    return similarity

def predict(user, item, ratings, similarity):
    if item not in ratings.columns or user not in ratings.index:
        return np.nan

    rated_by = ratings[item].dropna()
    similarity_scores = similarity.loc[user, rated_by.index].dropna()

    user_mean = ratings.loc[user].mean()
    neighbor_means = ratings.loc[similarity_scores.index].mean(axis=1)

    numerator = np.sum(similarity_scores * (rated_by - neighbor_means))
    denominator = np.sum(np.abs(similarity_scores))

    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return np.nan

    prediction = user_mean + numerator / denominator
    return prediction

def group_predictions(group, ratings, similarity):
    all_items = ratings.columns
    group_preds = pd.DataFrame(index=group, columns=all_items)

    for user in group:
        for item in all_items:
            pred = predict(user, item, ratings, similarity)
            group_preds.loc[user, item] = pred

    return group_preds

def aggregate(preds, method='average'):
    if method=='average':
        return preds.mean(axis=0)
    elif method=='least misery':
        return preds.min(axis=0)
    else:
        return

def disagreement_aggregate(preds, alpha=0.5):
    avg_scores = preds.mean(axis=0)
    std_scores = preds.std(axis=0).fillna(0)
    # penalizing items with higher disagreement
    adjusted_scores = avg_scores - alpha * std_scores
    return adjusted_scores

def main():
    # a) Display first 5 rows, and number of rows
    print("a) example rows of data and row amount")
    df = pd.read_csv('ratings.csv')
    print(df.head())
    print(len(df))

    # b) 
    #Builds user-item matrix, and computes user similarity using peasrson correlation
    ratings_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
    user_similarity = pearson(ratings_matrix)
    print("b) similarity table with Pearson")
    print(user_similarity.round(3))
     
    #Choosing a user and unrated movie, and preparing for prediction
    user = int(input("Enter userId (1 - 610): "))
    user_ratings = ratings_matrix.loc[user]
    rated_items = user_ratings[user_ratings.notna()].index.tolist()
    item = int(input(f"Enter moveId from unrated items: "))
    if item in rated_items:
        print(f"Item is already rated by user {user}!")
        exit()
    
    #Predicting a rating user would give to a movie based on pearson correlation
    pred = predict(user, item, ratings_matrix, user_similarity)
    print(f"c) prediction with Pearson: {pred:.3f}")

    # d) let's use Kendall Tau correlation (also suitable for pairwise correlation) 
    # Calculating user similarity using Kendall Tau correlation and displaying the similarity table.
    user_similarity = kendall(ratings_matrix)
    print("d) similarity table with Kendall Tau")
    print(user_similarity.round(3))

    pred = predict(user, item, ratings_matrix, user_similarity)
    print(f"\nPrediction with Kendall Tau {pred:.3f}\n")

    # e) 
    group_input = input("Enter users for group (e.g. 1 2 3): ")
    group = [int(u) for u in group_input.strip().split()]
    group_preds = group_predictions(group, ratings_matrix, user_similarity)

    # average aggregation
    aggregated = aggregate(group_preds)
    recommended_items = aggregated.sort_values(ascending=False).dropna()
    print(f"e) recommended items for group {group} with average aggregation:")
    print(recommended_items)
    print("\n")

    # least misery aggregation
    aggregated = aggregate(group_preds, method='least misery')
    recommended_items = aggregated.sort_values(ascending=False).dropna()
    print(f"recommended items for group {group} with least misery aggregation:")
    print(recommended_items)
    print("\n")

     # f) disagreement aggregation (it downranks movies that split the group, even if some people love a movie, if others hate it that movie is pushed down the list)
    aggregated_disagree = disagreement_aggregate(group_preds, alpha=0.5)
    recommended_items_disagree = aggregated_disagree.sort_values(ascending=False).dropna()
    print(f"f)recommended items for group {group} with disagreement aggregation:")
    print(recommended_items_disagree)
    print("\n")
if __name__=="__main__":
    main()