# Data Mining Projects

Collection of data mining and machine learning projects, exploratory analyses, and model evaluations.

## 1. Online Retail Data Mining

Data mining, exploratory analysis, classification, and market basket analysis on the Online Retail dataset (`online_retail.csv`, `ISO-8859-1` encoding).

### Files
- `Online_Retail_Data_Mining.ipynb` — Original Jupyter / Google Colab notebook with code and outputs.
- `online_retail_data_mining.py` — Standalone Python script extracted from the notebook.

### Workflow Overview
1. **Data Loading & Cleaning**: Drops duplicates, removes missing `Description` rows, filters out cancelled invoices (`InvoiceNo` starting with `C`), keeps positive `Quantity` and `UnitPrice`, and computes `TotalAmount`.
2. **Exploratory Data Analysis**: Top 10 selling products by quantity, top 10 countries by total sales, and quantity distribution histogram.
3. **Order Classification**: Aggregates transactions by `InvoiceNo`, labels orders as High Value (`1`) or Low Value (`0`) relative to median `TotalAmount`, and trains a `DecisionTreeClassifier` (`criterion="entropy"`, `max_depth=5`).
4. **Market Basket Analysis**: Filters `United Kingdom` transactions across the top 30 products, mines frequent itemsets with `apriori` (`min_support=0.01`, `max_len=3`), and generates association rules (`metric="confidence"`, `min_threshold=0.30`).

---

## 2. UCI Heart Disease Prediction (`data mining project 2/`)

Classification and model comparison on the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features, target `num`).

### Files
- `data mining project 2/UCI_Heart_Disease_Colab.ipynb` — Structured Colab notebook with preprocessing, repeated stratified cross-validation, hyperparameter tuning, and evaluation.
- `data mining project 2/heart_disease_data_set.ipynb` — Executed notebook with generated plots and outputs.
- `data mining project 2/uci_model_comparison_results.csv` — Hold-out test metrics comparing Decision Tree, Naive Bayes, KNN, and ANN (MLP).

---

## Setup & Usage

```bash
pip install -r requirements.txt
python online_retail_data_mining.py
```
