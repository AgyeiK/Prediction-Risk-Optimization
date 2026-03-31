# 🚀 Data Science Portfolio: Prediction & Risk Optimization
**Project Showcase by Kwabena Agyei**

This repository features two high-impact machine learning projects. These projects demonstrate my ability to handle both **high-precision regression** for financial forecasting and **safety-critical classification** for public health.

---

## 🍄 Project 1: #1 Ranked Mushroom Safety AI (Classification)
**Objective:** Classify mushrooms as edible or poisonous with a strict **"Zero-Hospitalization"** mandate.

### 🏆 The Winning Strategy
In this challenge, the cost of a **False Negative** (labeling a poisonous mushroom as "edible") is a critical failure. While most standard models optimize for raw Accuracy, I pivoted to a **Recall-optimized** strategy to ensure 100% safety.

* **Algorithm:** **XGBoost Classifier** with 1,000 estimators and a 0.01 learning rate for surgical precision.
* **Robust Encoding:** Implemented a **Global Label Encoding** strategy, fitting the encoder on a concatenated view of Train and Test data to prevent "Unknown Label" crashes during inference.
* **Probability Thresholding:** * I bypassed the standard `0.5` binary cutoff.
    * Implemented a **`0.05` Safety Threshold** (if there is a 5% chance of toxicity, it is flagged as poisonous).
    * **Result:** This eliminated the 25 hospitalizations seen in baseline models.

### 📊 Performance Summary
| Metric | Result | Note |
| :--- | :--- | :--- |
| **Hospitalizations** | **0** | **Target Achieved** |
| **Recall (Safety Score)** | **1.00** | Caught 100% of poisonous threats |
| **Leaderboard Rank** | **#1** | Outperformed the competition |
| **Final Scoring Metric** | **0.9972** | Maximized score by removing penalties |

---

## 🏠 Project 2: Ames Housing Price Predictor (Regression)
**Objective:** Predict residential home prices in Ames, Iowa, using a complex set of 79 explanatory variables.

### 🛠️ Technical Implementation
* **Target Transformation:** Utilized `np.log1p` on the `SalePrice`. This normalizes right-skewed price data, ensuring the model remains accurate across both budget and luxury properties.
* **Ensemble Blending:** Developed a weighted blend of **Gradient Boosting** and **Ridge Regression**.
    * *Gradient Boosting* captures non-linear trends and feature interactions.
    * *Ridge Regression* provides L2 regularization to prevent overfitting in high-dimensional space.
* **Feature Alignment:** Automated the alignment of training and testing features to ensure zero-error inference.

### 📈 Key Insights
* **Ensemble Advantage:** Demonstrated that blending models with different biases (Linear vs. Tree-based) produces a more stable and accurate prediction than any single model.
* **Data Integrity:** Handled missing values and categorical encoding strategically to maintain high stability across cross-validation folds.

---

## 💻 Tech Stack & Tools
* **Languages:** Python (Pandas, NumPy)
* **ML Libraries:** XGBoost, Scikit-Learn
* **Preprocessing:** Label Encoding, Target Encoding, Log-Scaling
* **Optimization:** Custom Probability Thresholding, Ensemble Blending

---

## 📂 Project Structure
```bash
├── Mushroom_Safety/
│   ├── mushroom_classifier.ipynb   # #1 Ranked Logic (XGBoost)
│   └── submission_final.csv        # 0 Hospitalization output
├── Ames_Housing/
│   ├── housing_regression.ipynb    # Ensemble Blend Logic
│   └── housing_preds.csv           # Regression output
└── README.md                       # Portfolio Overview
