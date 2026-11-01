# Hospital Predictive Analysis
# Predicting hospital bed capacity using five classification models

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv(r"C:\Users\risha\Downloads\hospital_directory.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Select required columns
cols = ["Total_Num_Beds", "Number_Doctor", "Num_Mediconsultant_or_Expert",
        "Establised_Year", "Hospital_Category", "Hospital_Care_Type",
        "Discipline_Systems_of_Medicine", "State", "Emergency_Services",
        "Tariff_Range"]
data = df[cols].copy()

# 3. Clean data
data = data.drop_duplicates()
data["Total_Num_Beds"] = pd.to_numeric(data["Total_Num_Beds"], errors="coerce")
data = data.dropna(subset=["Total_Num_Beds"])

num_cols = ["Number_Doctor", "Num_Mediconsultant_or_Expert", "Establised_Year"]
cat_cols = ["Hospital_Category", "Hospital_Care_Type", "Discipline_Systems_of_Medicine",
            "State", "Emergency_Services", "Tariff_Range"]

for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# 4. Create three bed-capacity classes
data["Bed_Category"] = pd.qcut(
    data["Total_Num_Beds"].rank(method="first"),
    3,
    labels=["Low", "Medium", "High"]
)
print("\nBed Categories:\n", data["Bed_Category"].value_counts())

# 5. Basic data visualization
data["Hospital_Category"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Hospital Categories")
plt.xlabel("Category")
plt.ylabel("Number of Hospitals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data["Bed_Category"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Hospital Bed Capacity")
plt.ylabel("")
plt.show()

data["Total_Num_Beds"].plot(kind="hist", bins=25)
plt.title("Distribution of Total Beds")
plt.xlabel("Total Beds")
plt.show()

plt.scatter(data["Number_Doctor"], data["Total_Num_Beds"])
plt.title("Doctors vs Total Beds")
plt.xlabel("Number of Doctors")
plt.ylabel("Total Beds")
plt.show()

# 6. Prepare data for models
X = data[num_cols + cat_cols]
X = pd.get_dummies(X, columns=cat_cols)
y = data["Bed_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scaling is used for KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Create exactly five syllabus models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# 8. Train and evaluate each model
for name, model in models.items():
    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    print("\nModel:", name)
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(pre, 3))
    print("Recall:", round(rec, 3))
    print("F1 Score:", round(f1, 3))
    print(classification_report(y_test, pred, zero_division=0))

    results.append([name, acc, pre, rec, f1])

    # Confusion matrix for each model
    cm = confusion_matrix(y_test, pred)
    plt.imshow(cm, cmap="Blues")
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()

# 9. Compare the five models
result = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nModel Comparison:\n", result.sort_values("Accuracy", ascending=False))

result.plot(x="Model", y="Accuracy", kind="bar", legend=False)
plt.title("Accuracy Comparison of Five Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 10. Decision Tree graph
plt.figure(figsize=(14, 6))
plot_tree(models["Decision Tree"], max_depth=3, filled=True,
          feature_names=X.columns, class_names=["Low", "Medium", "High"])
plt.title("Decision Tree")
plt.show()

# 11. Random Forest feature importance
importance = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
