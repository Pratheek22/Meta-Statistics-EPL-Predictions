## EPL Match Outcome Prediction: Gaining a Statistical Edge using Machine Learning

This repository contains the source code and notebooks on predicting the outcome (Win, Draw, or Loss) of English Premier League (EPL) soccer matches.

---

## 1. Project Goal

The primary objective was to evaluate whether various machine learning models (SoftMax Logistic Regression, Neural Network, GDA, and SVM) could accurately predict the multi-class outcome of EPL matches using pre-match team and player **meta statistics**.

---

## 2. Key Findings

Model Performance Summary

1. Softmax Logistic Regression (Multi-class: Win/Draw/Loss)

Test Accuracy: 55.34% – the highest among all multi-class models
Successfully handled class imbalance better than baseline and other approaches
Consistently outperformed other multi-class algorithms
Challenge: The 'Draw' prediction remains the hardest class for all algorithms

2. Neural Network - MLPClassifier (Multi-class: Win/Draw/Loss)

Test Accuracy: 54.68%
Overfitting Gap: ~0.34 (significant)
Training accuracy ranged from 0.77–0.89, but generalization was poor
The large gap between training and test performance indicates learned patterns did not transfer well
Model instability attributed to:

Architecture complexity
Limited data availability



3. Support Vector Machine (SVM) Results

Best performing kernel: Polynomial (degree=3) with 0.5863 accuracy on Linear kernel
Kernel comparison:

GDA: 0.585
Linear: 0.580
RBF: 0.584
Poly: 0.588 (d=3)


Performance across kernels was relatively consistent, with marginal differences


4. Logistic Regression (Binary: Win/Loss only)

Test Accuracy: 76.36% – dramatically higher than multi-class models
Precision, Recall, F1-Score: All approximately 0.55
Demonstrates that removing 'Draw' class substantially improves prediction reliability
Confirms the 'Draw' class is the primary constraint on multi-class accuracy


**Core Finding
The single largest barrier to prediction accuracy is the inherent ambiguity of the 'Draw' result.**

Draw outcomes are extremely difficult to distinguish from marginal Win/Loss cases across all model architectures
The 20+ percentage point improvement when predicting binary outcomes (Win/Loss only: 76.36%) versus multi-class (Win/Draw/Loss: ~55%) quantifies this challenge
This suggests that soccer match draws exist in a "decision boundary overlap zone" where feature patterns are highly similar to close wins or losses

Regularization, cross-validation, and data augmentation are recommended for improvement.

---

## 3. Repository Contents

| File Name | Description | Models Covered |
| :--- | :--- | :--- |
| `Logistic_regression_with_tie.ipynb` | **Primary Notebook.** Multi-class SoftMax LogReg implementation and best result. | SoftMax LogReg |
| `Logistic_regression.ipynb` | Experiment notebook for binary classification (Draws removed). | LogReg (Binary) |
| `Neural_Net.ipynb` | Implementation and tuning of the Neural Network (MLPClassifier). | Neural Network |
| `Softmax_Gaussian_SVM_F1.ipynb` | Notebook exploring GDA and SVM performance with Feature Set 1 (Basic). | GDA, SVM |
| `Softmax_Gaussian_SVM-F2.ipynb` | Notebook exploring GDA and SVM performance with Feature Set 2. | GDA, SVM |
| `Softmax_Gaussian_SVM-F3.ipynb` | Notebook exploring GDA and SVM performance with Feature Set 3. | GDA, SVM |

---

## 4. Setup and Execution Instructions

### A. Prerequisites

This project requires a Python environment with the following libraries:

* `numpy`
* `matplotlib`
* `scikit-learn` (`sklearn`)
* `seaborn`
* `jupyter` (or `ipython` to run the notebooks)

Install all necessary libraries using `pip`:

```bash
pip install numpy matplotlib scikit-learn seaborn jupyter

### B. Running the Project

1.  **Clone the Repository:**
    Start by cloning the project files from your GitHub repository to your local machine:
    ```bash
    git clone (https://github.com/Pratheek22/EPL_predictions)
    cd EPL_predictions
    ```

2.  **Verify Data:**
    Ensure the main dataset file (e.g., `match_vectors_extended.csv` or `match_vectors_no_label.csv`) is present in the project's directory (same as folder structure in this repo) , as the notebooks rely on these files.

3.  **Launch Jupyter:**
    Start the Jupyter environment from the project's root directory:
    ```bash
    jupyter notebook
    ```

4.  **Execute Notebooks:**
    Open the notebooks in your browser and run the cells in sequence (e.g., **Cell 1** to **Cell Last**).
    * To see the best multi-class result, run `Logistic_regression_with_tie.ipynb`.
    * To see the binary experiment result, run `Logistic_regression.ipynb`.
    * To see the Neural Network analysis, run `Neural_Net.ipynb`.
    * Same with all other .ipynb files.

The final output of each notebook will print the **confusion matrix** and the **final test accuracy** directly to the notebook's output cells.
