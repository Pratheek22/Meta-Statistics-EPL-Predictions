# ⚽ Gaining a Statistical Edge in Soccer Prediction using Machine Learning

This project applies **machine learning techniques** to predict English Premier League (EPL) match outcomes using a variety of **meta-statistics** including player, team, and tactical data. It explores how different feature sets impact predictive accuracy and identifies the role of simplicity vs. complexity in performance.

---

## 📘 Project Overview

Predicting soccer results is challenging due to the dynamic nature of player performance and tactics. Traditional models relying on historical results often fail to capture crucial contextual factors like squad composition and formation. This project focuses on:

* Leveraging **meta-statistics** (aggregated player and team data)
* Comparing different feature sets (overall ratings, positional averages, individual metrics)
* Evaluating multiple ML models for classification accuracy

---

## 🧩 Dataset

* **Source:** [WhoScored.com](https://www.whoscored.com/)
* **Seasons:** 2009–2019 (3,800 matches)
* **Features:**

  * **F1:** Overall team ratings (2 features)
  * **F2:** Positional averages (10 features)
  * **F3:** Detailed player statistics (280 features)
* **Split:** 80% training / 20% testing
* **Baseline:** 46.19% home-win rate

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression

* Multinomial and dual-binary variants (with tie handling)
* Meta logistic regression combining probability outputs
* Custom gradient descent implementation with L2 regularization

**Best Accuracy:** 55.34% (Full feature set with tie handling)

### 2️⃣ Neural Network (MLP)

* Single hidden layer, ReLU activation
* Hidden nodes varied (9–19), best at 14
* Standardized inputs for stable training

**Best Accuracy:** 54.68% (14 hidden nodes)

### 3️⃣ Support Vector Machine (SVM)

* Gaussian, Linear, and Polynomial kernels
* Softmax-SVM hybrid architecture for probabilistic classification
* Polynomial (degree 3) achieved best results

**Best Accuracy:** 58.8% (Polynomial kernel, F1 feature set)

---

## 📊 Results Summary

| Model               | Feature Set | Accuracy  | Key Notes                                          |
| :------------------ | :---------- | :-------- | :------------------------------------------------- |
| Logistic Regression | F3          | 55.34%    | Best with tie-handling approach                    |
| Neural Network      | F3          | 54.68%    | Overfitting due to limited data                    |
| SVM                 | F1          | **58.8%** | Polynomial kernel, non-linear feature interactions |

---

## 🔍 Key Findings

* **Simplicity Wins:** Overall team ratings outperform complex player-level data.
* **Overfitting:** High-dimensional F3 features degrade model generalization.
* **Non-linearity Matters:** Polynomial kernel best captures team interactions.
* **Draw Prediction Challenge:** Low recall (~25–35%) due to overlapping features.
* **Performance Ceiling:** Pre-match accuracy caps at ~60% for EPL data.

---

## 🧠 Conclusion

The study demonstrates that **parsimonious models** leveraging aggregated team metrics outperform deeper models when data is limited. Simpler features (overall ratings) provided better generalization than granular player data. Future improvements include ensemble methods, draw-specific classification, and expanding to multi-league datasets.

---

## 🛠️ Tools & Libraries

* Python (NumPy, Pandas, scikit-learn, Matplotlib)
* Jupyter Notebooks
* Data sourced via web scraping (WhoScored API)

---

## 🧑‍💻 Authors

**Pratheek J Gowda** - PES1UG23CS448
**Rachana R** - PES1UG23CS459

---

