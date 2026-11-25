import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("All libraries imported successfully")

url = "https://github.com/dbdmg/data-science-lab/raw/master/datasets/online_retail.csv"

df = pd.read_csv(url, encoding="ISO-8859-1")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

df.head()

print(df.head())


print(df.info())


print(df.describe())


print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()

df = df.dropna(subset=["Description"])

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("Data cleaning completed")
print(df.head())
print("Cleaned dataset shape:", df.shape)

top_products = df.groupby("Description")["Quantity"].sum().nlargest(10)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")
plt.title("Top 10 Selling Products")
plt.xlabel("Product Name")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

country_sales = df.groupby("Country")["TotalAmount"].sum().nlargest(10)

plt.figure(figsize=(10,5))
country_sales.plot(kind="bar")
plt.title("Top 10 Countries by Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Quantity"], bins=30)
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Quantity"], bins=30)
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

orders = df.groupby("InvoiceNo").agg(
    TotalAmount=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    AveragePrice=("UnitPrice", "mean"),
    Country=("Country", "first")
).reset_index()


median_sales = orders["TotalAmount"].median()

orders["OrderType"] = (
    orders["TotalAmount"] > median_sales
).astype(int)



print(orders.head())
print(orders["OrderType"].value_counts())

X = orders[
    ["TotalQuantity", "UniqueProducts", "AveragePrice"]
]

y = orders["OrderType"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree model trained successfully")

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

print(classification_report(y_test, y_pred))

plt.figure(figsize=(18,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Low Value", "High Value"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Classification")
plt.show()

retail = df[df["Country"] == "United Kingdom"].copy()

top_30 = retail["Description"].value_counts().head(30).index

retail = retail[retail["Description"].isin(top_30)]

basket = retail.groupby(
    ["InvoiceNo", "Description"]
)["Quantity"].sum().unstack(fill_value=0)

basket_encoded = basket.gt(0)

print("Transaction matrix:")
print(basket_encoded.head())

frequent_itemsets = apriori(
    basket_encoded,
    min_support=0.01,
    use_colnames=True,
    max_len=3
)

print("Frequent Itemsets:")
print(frequent_itemsets.sort_values(
    "support", ascending=False
).head(15))

rules = association_rules(
    frequent_itemsets,
    num_itemsets=len(basket_encoded),
    metric="confidence",
    min_threshold=0.30
)

rules = rules.sort_values(
    "lift", ascending=False
)

print("Association Rules:")
print(rules[
    ["antecedents", "consequents",
     "support", "confidence", "lift"]
].head(10))

print("========== FINAL RESULTS ==========")

print("Total Transactions:", df["InvoiceNo"].nunique())

print("Total Products:", df["Description"].nunique())

print("Decision Tree Accuracy:",
      round(accuracy * 100, 2), "%")

print("Frequent Itemsets Found:",
      len(frequent_itemsets))

print("Association Rules Found:",
      len(rules))

print("===================================")
