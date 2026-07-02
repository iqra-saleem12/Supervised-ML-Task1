# Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Regression Dataset
from sklearn.datasets import fetch_california_housing

print("Libraries Imported Successfully!")
# Load Dataset

df = pd.read_csv("dataset/Housing.csv")

print("Housing Dataset Loaded Successfully!")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
# Data Cleaning

print("\nChecking Missing Values:")
print(df.isnull().sum())

print("\nRemoving Duplicate Rows...")

df.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully!")

print("\nDataset Shape After Cleaning:")
print(df.shape)
# NumPy Operations

price = np.array(df["price"])

print("\nAverage House Price:", np.mean(price))
print("Maximum House Price:", np.max(price))
print("Minimum House Price:", np.min(price))
# Pandas Operations

print("\nAverage Price by Bedrooms:")
print(df.groupby("bedrooms")["price"].mean())

print("\nAverage Price by Bathrooms:")
print(df.groupby("bathrooms")["price"].mean())
# House Price Distribution

plt.figure(figsize=(8,5))

plt.hist(df["price"], bins=20, color="skyblue", edgecolor="black")

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")

plt.savefig("images/price_distribution.png")
plt.close()

print("Price Distribution Graph Completed!")
# Area vs Price

plt.figure(figsize=(8,5))

sns.scatterplot(data=df, x="area", y="price")

plt.title("Area vs Price")

plt.savefig("images/area_vs_price.png")
plt.close()

print("Area vs Price Graph Completed!")
# Correlation Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.savefig("images/correlation_heatmap.png")
plt.close()

print("Heatmap Completed!")
# Convert Yes/No columns into 1/0

yes_no_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in yes_no_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})

# Convert furnishingstatus into dummy variables

df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

print("\nData Preprocessing Completed!")
print(df.head())
# Train-Test Split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features and Target

X = df.drop("price", axis=1)
y = df["price"]

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Linear Regression Model

model = LinearRegression()

print("Training Linear Regression Model...")

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Prediction

y_pred = model.predict(X_test)

# Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Ridge Regression

from sklearn.linear_model import Ridge

ridge = Ridge()

print("\nTraining Ridge Regression Model...")

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, ridge_pred)
mse = mean_squared_error(y_test, ridge_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, ridge_pred)

print("\nRidge Regression Results")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Lasso Regression

from sklearn.linear_model import Lasso

lasso = Lasso()

print("\nTraining Lasso Regression Model...")

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

mae = mean_absolute_error(y_test, lasso_pred)
mse = mean_squared_error(y_test, lasso_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, lasso_pred)

print("\nLasso Regression Results")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)

print("\nTraining Decision Tree Regressor...")

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

mae = mean_absolute_error(y_test, dt_pred)
mse = mean_squared_error(y_test, dt_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree Results")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\nTraining Random Forest Regressor...")

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, rf_pred)
mse = mean_squared_error(y_test, rf_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
from sklearn.model_selection import cross_val_score

print("\nCross Validation (Linear Regression)")

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation Scores:", cv_scores)
print("Average R2 Score:", cv_scores.mean())
from sklearn.model_selection import GridSearchCV

print("\nHyperparameter Tuning (Random Forest)...")

parameters = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=parameters,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Score:", grid.best_score_)
# Regression Model Comparison

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, ridge_pred),
        r2_score(y_test, lasso_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred)
    ]
})

print("\nRegression Model Comparison")
print(comparison)
plt.figure(figsize=(8,5))

plt.bar(comparison["Model"], comparison["R2 Score"])

plt.title("Regression Model Comparison")
plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.xticks(rotation=20)

plt.savefig("regression_model_comparison.png")

plt.show()

print("Regression Comparison Graph Saved Successfully!")
# ==========================================
# CLASSIFICATION DATASET (TITANIC)
# ==========================================

print("\n==============================")
print("TITANIC CLASSIFICATION DATASET")
print("==============================")

titanic = pd.read_csv("dataset/titanic.csv")

print("\nTitanic Dataset Loaded Successfully!")

print(titanic.head())

print("\nDataset Shape:")
print(titanic.shape)

print("\nDataset Information:")
print(titanic.info())

print("\nMissing Values:")
print(titanic.isnull().sum())

print("\nSummary Statistics:")
print(titanic.describe())
print("\nCleaning Titanic Dataset...")

# Fill missing Age values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())

# Fill missing Embarked values
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])

# Drop Cabin column
titanic.drop("Cabin", axis=1, inplace=True)

print("Titanic Dataset Cleaned Successfully!")

print("\nMissing Values After Cleaning:")
print(titanic.isnull().sum())
print("\nTitanic Data Preprocessing...")

# Convert Gender to Numbers
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})

# Convert Embarked to Numbers
titanic = pd.get_dummies(titanic, columns=["Embarked"], drop_first=True)

# Drop unnecessary columns
titanic.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print("Titanic Preprocessing Completed!")

print(titanic.head())
# ===============================
# Feature Selection
# ===============================

X = titanic.drop("Survived", axis=1)
y = titanic["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTitanic Train-Test Split Completed!")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature Scaling Completed!")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nTraining Logistic Regression Model...")

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("Model Trained Successfully!")

print("\nLogistic Regression Results")

print("Accuracy:", accuracy_score(y_test, log_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, log_pred))

print("\nClassification Report")
print(classification_report(y_test, log_pred))
from sklearn.neighbors import KNeighborsClassifier

print("\nTraining KNN Model...")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("\nKNN Results")

print("Accuracy:", accuracy_score(y_test, knn_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, knn_pred))

print("\nClassification Report")
print(classification_report(y_test, knn_pred))
from sklearn.naive_bayes import GaussianNB

print("\nTraining Naive Bayes Model...")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Results")

print("Accuracy:", accuracy_score(y_test, nb_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, nb_pred))

print("\nClassification Report")
print(classification_report(y_test, nb_pred))
from sklearn.tree import DecisionTreeClassifier

print("\nTraining Decision Tree Model...")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Results")

print("Accuracy:", accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))
from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))
from sklearn.svm import SVC

print("\nTraining SVM Model...")

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("\nSVM Results")

print("Accuracy:", accuracy_score(y_test, svm_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, svm_pred))

print("\nClassification Report")
print(classification_report(y_test, svm_pred))
from sklearn.model_selection import cross_val_score

print("\nCross Validation (Logistic Regression)")

cv_scores = cross_val_score(log_model, X_train, y_train, cv=5)

print("Cross Validation Scores:", cv_scores)

print("Average Accuracy:", cv_scores.mean())
from sklearn.model_selection import GridSearchCV

print("\nHyperparameter Tuning (Random Forest)...")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Score:", grid.best_score_)
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred)
    ]
})

print("\nClassification Model Comparison")
print(comparison)

plt.figure(figsize=(8,5))

plt.bar(comparison["Model"], comparison["Accuracy"])

plt.xticks(rotation=20)

plt.title("Classification Model Comparison")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("images/classification_model_comparison.png")

plt.show()

print("Classification Comparison Graph Saved Successfully!")