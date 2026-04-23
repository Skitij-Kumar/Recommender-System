
# 🎬Recommender Systems



## 📄Overview
 I developed a multi-part group recommendation system. The project covers collaborative filtering, novel algorithm design (WIAA), sequential group recommendations using MMR, and counterfactual explainability in 4 different parts. 
## 📊Dataset
- **Source:** MovieLens ratings dataset  
- **Users:**  16 users (Alisson, Simon, Michael, Dani, Brian, David, Ilan, Juan, Julius, Philipp, Joshua, Rick, Moises, Fabian, Isaac, Kurt)
## 🛠️ Tech Stack
- **Tools used:** Python, Pandas, NumPy, Jupyter Notebook
- **Technique / Methods used:** 
  - Pearson and Kendall Tau correlation for similarity
  - Average, Least Misery, and Disagreement Aggregation
  - WIAA (Weighted Individual Agreement Aggregation)
  - MMR (Maximal Marginal Relevance)
  - Cosine Similarity on genre matrices
  - Greedy Grow method
  - Counterfactual Explanation
## 🚀Steps / Workflow
1. 🔍 **Part 1 - Collaborative Filtering:** 
- Built user-item matrix from ratings dat
- Computed Pearson correlation between users to measure linear similarity
- Implemented Kendall Tau similarity using order of preference instead of absolute rank values
- Built prediction function using weighted neighbor deviations formula
- **Implemented group recommendations using:**
-  Average Aggregation (used to maximize overall group satisfaction)
-  Least Misery Aggregation (used to pick movies no one dislikes)
-  Disagreement Aggregation (used to penalize movies with high standard deviation across group ratings)
2.  🔍 **Part 2 - WIAA (Novel Algorithm):**
- Studied existing SQUIRREL Framework strategies: SDAA, and SIAA
- Developed WIAA (Weighted Individual Agreement Aggregation) as a novel extension of SIAA
- Replaced SIAA's userDis with userAlign, instead of measuring how far behind a user is from the happiest person, WIAA measures how similar a user's ratings are to what the group historically chooses
- userAlign formula penalizes persistent polarisation and rewards users who align with group taste group taste
- Implemented iterative weight updating across recommendation rounds using lambda and rehab factor parameters
3.  🔍 **Part 3 - Sequential Group Recommendations with MMR:**
- Combined WIAA with MMR (Maximal Marginal Relevance) for sequential multi-round recommendations
- Created genre matrix for all movies using one-hot encoding
- Computed cosine similarity between movies to measure genre overlap
- MMR scoring: Score = relevance − similarity to selected − similarity to history
- Updated recommendation history each round to avoid repeating similar movies
4.  🔍 **Part 4 - Greedy Grow and Counterfactual Explainability:**
- Built Greedy Grow method to select top-K movies iteratively based on average group scores
- Predicted missing ratings using cosine similarity between users
- Final group score = average of real and predicted ratings combined
- **Implemented counterfactual explanation system:**
- Step 1: Removed single items to check if target movie disappears from top-K
- Step 2: If no single item worked, I tried removing pairs of items
- Step 3: If neither works, recommendation is classified as stable
## 🎯Results
- Final output format: Movie ID, Predicted Group Rating, Counterfactual Explanation
- Example: 593, 4.71, explanation: None → stable recommendation not influenced by any single or pair of items
- WIAA successfully balances group fairness by dynamically adjusting user weights based on historical alignment
- MMR ensures diverse recommendations across multiple rounds by penalising genre repetition
## 🛠️How to Run the Project
Step 1- Donload python code parts folder (e.g., project part-1, project part-2, etc.)

Step 2 - Run it in Jupyter notebook or text editor of your choice

**}**
## 🔗Demo
- Watch Video of Car Sales Analsis wit Excel: (https://drive.google.com/file/d/12MTsZhqvh38RzZvnLFMA0WChfYSacvgQ/view?usp=sharing)

- Here’s a preview of the interactive dashboard created with Cognos:

Car Sales Dashboard
![Car sales dashboard Screenshot](https://github.com/Skitij-Kumar/All-Data-Projects/blob/main/Car-Sales-Analysis-and-Visualization-Project/Main%20project%20files/Car%20sales%20dashboard%20Cognos%20Image.png)
Cars Service Dashboard
![Cars service](https://github.com/Skitij-Kumar/All-Data-Projects/blob/main/Car-Sales-Analysis-and-Visualization-Project/Main%20project%20files/Cars%20service%20dashboard%20Cognos%20Image.png)


## 🤝Contact
💼 https://www.linkedin.com/in/skitij-kumar/ | 📧 Skitijkumar24@gmail.com
