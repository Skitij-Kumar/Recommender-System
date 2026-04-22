
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
-  Average Aggregation, used to maximize overall group satisfaction
-  Least Misery Aggregation, used to pick movies no one dislikes
-  Disagreement Aggregation, used to penalize movies with high standard deviation across group ratings
2. 📈 **Dashboarding using Cognos for sales (1st Dashboard):**
- Visualized profit made by dealers and quanity sold by car models using KPIs, bar, and column chart.
3. 📈 **Dashboarding using Cognos for services (2nd Dashboard):**
- Visualized recalls per car model and most affected systems using column chart and heat map.
- Analyzed customer sentiments in reviews with a tree map.
## 🎯Results
- The dealer with ID 1288 generated the highest profit.
- The Hudson model was purchased more frequently than others.
- The Beaufort model got more services or inspections.
- The airbag in the Beaufort model failed most frequently.
## 🛠️How to Run the Project
### For Excel files
Step 1- Open main project file folder.

Step 2- Download the excel car sales analysis and dataset folder files and open the downloaded files in excel (or you can just watch 1 minute video using the given link in Demo section).


### For IBM Cognos dashboard
Step 1- Double click on the pdf or png file to see dashboards.

Step 2- In pdf file scroll down to see second dashboard.


**{Note:** 
- I have used the IBM Cognos free trial version, and it has expired. Unfortunately, I am unable to provide a link of a dashboard, even though I would really like to share it and show it in Cognos.
- Excel file contains 5 excel worksheets inside it.

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
