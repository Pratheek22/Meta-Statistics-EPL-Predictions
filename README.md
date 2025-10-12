## EPL Match Outcome Prediction: Gaining a Statistical Edge using Machine Learning

This repository contains the source code and notebooks on predicting the outcome (Win, Draw, or Loss) of English Premier League (EPL) soccer matches.

---

## 1. Project Goal

The primary objective was to evaluate whether various machine learning models (SoftMax Logistic Regression, Neural Network, GDA, and SVM) could accurately predict the multi-class outcome of EPL matches using pre-match team and player **meta statistics**.

Our core target was to **surpass the established literature benchmark of approximately 53.25% accuracy**.

---

## 2. Key Findings

| Model | Task | Test Accuracy | Note |
| :--- | :--- | :--- | :--- |
| **SoftMax Logistic Regression** | Multi-class (Win/Draw/Loss) | **55.34%** | Best-performing multi-class model. |
| **Neural Network (MLPClassifier)** | Multi-class (Win/Draw/Loss) | **54.68%** | Achieved benchmark but struggled with overfitting. |
| **Logistic Regression** | Binary (Win/Loss Only) | **76.36%** | High score confirms the 'Draw' class is the major challenge. |

The single largest barrier to higher accuracy is the **ambiguity of the 'Draw' outcome**, which is statistically difficult for all models to separate from marginal Win/Loss results.

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
    git clone [Your-GitHub-Repository-Link]
    cd [Your-Repository-Name]
    ```

2.  **Verify Data:**
    Ensure the main dataset file (e.g., `match_vectors_extended.csv` or `match_vectors_no_label.csv`) is present in the project's root directory, as the notebooks rely on these files.

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

The final output of each notebook will print the **confusion matrix** and the **final test accuracy** directly to the notebook's output cells.
