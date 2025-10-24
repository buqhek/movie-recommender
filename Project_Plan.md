# 🎬 KNN Movie Recommender — Project Plan

This project uses **k-Nearest Neighbors (KNN)** to build a simple **movie recommender system** using the MovieLens dataset.  
The goal is to understand the complete workflow of a data science project — from data exploration to modeling and evaluation — using `pandas`, `scikit-learn`, and visualization libraries like `plotly`.

---

## 🧭 Milestone Roadmap

### ✅ Milestone 1 — Environment & Data Setup
**Goal:** Prepare a clean, reproducible workspace.

- [x] Run `./setup_project` to:
  - [x] Create the Python virtual environment (`env`)
  - [x] Download and unzip the MovieLens dataset (`dataset-ml-32m`)
- [x] Verify that all CSV files (`movies.csv`, `ratings.csv`, etc.) load correctly.
- [x] Read the `README.txt` in the dataset and summarize what each file represents.

📁 **Deliverable:** A working notebook (`EDA.ipynb`) that successfully loads and inspects the data.

---

### 📊 Milestone 2 — Exploratory Data Analysis (EDA)
**Goal:** Understand the dataset and identify useful features.

- [x] Merge `movies.csv` and `ratings.csv` on `movieId`.
- [x] Plot distributions:
  - [x] Ratings frequency histogram
  - [x] Average rating by genre
  - [x] Number of ratings per movie
- [x] Explore:
  - [x] Most common genres
  - [x] Highest rated movies
  - [ ] Patterns in user behavior

🧠 *Exploration Zone:*  
Experiment with `plotly.express` for interactive plots (hovering, filtering, zooming).

📁 **Deliverable:** Clear, labeled plots that summarize the data’s main characteristics.

---

### 🧹 Milestone 3 — Data Preprocessing for KNN
**Goal:** Prepare the data for similarity-based recommendations.

- [ ] Create a **user–movie rating matrix** (`users x movies`).
- [ ] Handle missing ratings (e.g., fill with 0, mean, or leave sparse).
- [ ] Normalize ratings (subtract each user’s mean rating).

🧠 *Exploration Zone:*  
Compare performance with and without normalization.

📁 **Deliverable:** A processed matrix ready for modeling (`user_movie_matrix.csv` or in-memory DataFrame).

---

### 🤝 Milestone 4 — Build the KNN Recommender
**Goal:** Implement user-based and item-based collaborative filtering.

- [ ] Use `sklearn.neighbors.NearestNeighbors`
  - [ ] Fit the model on the user–movie matrix
  - [ ] Try different distance metrics (`cosine`, `euclidean`)
- [ ] Build functions:
  - [ ] `get_similar_users(user_id, k)`
  - [ ] `get_similar_movies(movie_title, k)`
  - [ ] `recommend_movies_for_user(user_id, n=10)`

🧠 *Exploration Zone:*  
Compare **user-based** vs **item-based** KNN recommendation.

📁 **Deliverable:** A notebook section showing sample recommendations for different users.

---

### 📈 Milestone 5 — Evaluation & Interpretation
**Goal:** Measure model quality and draw insights.

- [ ] Split ratings into train/test sets (e.g., by user).
- [ ] Evaluate recommendations with:
  - [ ] RMSE (for numeric accuracy)
  - [ ] Precision@K or Recall@K (for recommendation quality)
- [ ] Compare results to a simple baseline (e.g., “top 10 highest-rated movies overall”).

📁 **Deliverable:** A short results summary explaining how accurate and meaningful your model is.

---

### 🧩 Milestone 6 — Extensions & Next Steps (Optional)
**Goal:** Add unique value and explore deeper learning.

Ideas:
- [ ] Add **content-based filtering** (e.g., genre or title similarity)
- [ ] Combine KNN with other algorithms (e.g., matrix factorization)
- [ ] Build a simple **Flask or Streamlit** web app for interactive recommendations
- [ ] Deploy your app on **Render**, **Railway**, or **Hugging Face Spaces**

---

## 🧠 Learning Objectives

By completing this project, you’ll learn to:
- Structure a real-world ML project from start to finish  
- Perform data cleaning and exploratory analysis  
- Understand user–item interactions  
- Apply and interpret KNN for recommendation tasks  
- Communicate findings through visualizations and summaries  

---

## ⚙️ Tech Stack

| Category | Tool/Library |
|-----------|---------------|
| Data Analysis | pandas, numpy |
| Visualization | seaborn, plotly.express, matplotlib |
| Machine Learning | scikit-learn |
| Environment | virtualenv / venv |
| Optional Deployment | Flask, React, Streamlit |

---

## 🏁 Completion Criteria

You’ll know you’re done when you can:
- Run the notebook end-to-end without errors  
- Generate meaningful movie recommendations for any user  
- Explain **how** and **why** your KNN model works  
- Visualize your dataset and model results clearly  

---

🎯 **Final Note:**  
Don’t rush. Focus on *understanding* why each step
