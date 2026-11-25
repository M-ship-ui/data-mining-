# Online Retail Data Mining

Data mining, exploratory analysis, classification, and market basket analysis on the Online Retail dataset.

## Project Files

- `Online_Retail_Data_Mining.ipynb` — Original Jupyter / Google Colab notebook with code and outputs.
- `online_retail_data_mining.py` — Standalone Python script extracted from the notebook.
- `requirements.txt` — Python package dependencies.

## Workflow Overview

1. **Data Loading**: Loads `online_retail.csv` (`ISO-8859-1` encoding).
2. **Data Cleaning**: Drops duplicates, removes missing `Description` rows, filters out cancelled invoices (`InvoiceNo` starting with `C`), keeps positive `Quantity` and `UnitPrice`, and computes `TotalAmount`.
3. **Exploratory Data Analysis**:
   - Top 10 selling products by quantity
   - Top 10 countries by total sales
   - Quantity distribution histogram
4. **Order Classification**:
   - Aggregates transactions by `InvoiceNo` (`TotalAmount`, `TotalQuantity`, `UniqueProducts`, `AveragePrice`, `Country`)
   - Labels orders as High Value (`1`) or Low Value (`0`) relative to the median order `TotalAmount`
   - Trains a `DecisionTreeClassifier` (`criterion="entropy"`, `max_depth=5`) with an 80/20 stratified split and evaluates accuracy, confusion matrix, classification report, and tree plot
5. **Market Basket Analysis**:
   - Filters `United Kingdom` transactions across the top 30 products
   - Mines frequent itemsets using `apriori` (`min_support=0.01`, `max_len=3`)
   - Generates association rules (`metric="confidence"`, `min_threshold=0.30`) sorted by lift

## Setup & Usage

```bash
pip install -r requirements.txt
python online_retail_data_mining.py
```
