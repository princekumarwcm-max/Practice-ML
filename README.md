# 🤖 Machine Learning Practice

A practical Machine Learning repository containing implementations, experiments, preprocessing techniques, algorithms, ensemble methods, and model evaluation using **Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn**.

The purpose of this repository is to understand Machine Learning concepts through **hands-on coding and practical examples**.

---

# 📌 Repository Structure

```text
machine-learning-practice/
│
├── README.md
├── requirements.txt
│
├── 01_Data_Preprocessing/
│   ├── 01_Missing_Values.ipynb
│   ├── 02_Duplicates.ipynb
│   ├── 03_Data_Encoding.ipynb
│   ├── 04_Feature_Scaling.ipynb
│   └── 05_Outlier_IQR.ipynb
│
├── 02_Exploratory_Data_Analysis/
│   ├── 01_Univariate_Analysis.ipynb
│   ├── 02_Bivariate_Analysis.ipynb
│   ├── 03_Multivariate_Analysis.ipynb
│   └── 04_Data_Visualization.ipynb
│
├── 03_Supervised_Learning/
│   │
│   ├── Regression/
│   │   ├── 01_Linear_Regression.ipynb
│   │   ├── 02_Multiple_Linear_Regression.ipynb
│   │   ├── 03_Polynomial_Regression.ipynb
│   │   └── 04_Ridge_Lasso_Regression.ipynb
│   │
│   └── Classification/
│       ├── 01_Logistic_Regression.ipynb
│       ├── 02_KNN.ipynb
│       ├── 03_Decision_Tree.ipynb
│       ├── 04_Random_Forest.ipynb
│       ├── 05_SVM.ipynb
│       └── 06_Naive_Bayes.ipynb
│
├── 04_Ensemble_Learning/
│   ├── 01_Bagging.ipynb
│   ├── 02_Boosting.ipynb
│   ├── 03_Max_Voting.ipynb
│   ├── 04_AdaBoost.ipynb
│   ├── 05_Gradient_Boosting.ipynb
│   ├── 06_XGBoost.ipynb
│   └── 07_Stacking.ipynb
│
├── 05_Model_Evaluation/
│   ├── 01_Confusion_Matrix.ipynb
│   ├── 02_Accuracy.ipynb
│   ├── 03_Precision_Recall_F1.ipynb
│   ├── 04_ROC_AUC.ipynb
│   └── 05_Cross_Validation.ipynb
│
├── 06_Feature_Engineering/
│   ├── 01_Feature_Selection.ipynb
│   ├── 02_Feature_Extraction.ipynb
│   └── 03_Dimensionality_Reduction.ipynb
│
├── 07_Hyperparameter_Tuning/
│   ├── 01_GridSearchCV.ipynb
│   └── 02_RandomizedSearchCV.ipynb
│
├── datasets/
│   └── sample_dataset.csv
│
└── images/
    ├── iqr_before.png
    ├── iqr_after.png
    └── voting_classifier.png
```

---

# 🧠 Machine Learning

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every individual task.

### Basic Machine Learning Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
EDA
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
Final Model
```

---

# 🛠️ Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/machine-learning-practice.git
```

Move into the project:

```bash
cd machine-learning-practice
```

Create virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

---

# 📄 requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# 1️⃣ Data Preprocessing

Data preprocessing is the process of converting raw data into a clean and suitable format for Machine Learning models.

Common preprocessing steps:

```text
Missing Value Handling
       ↓
Duplicate Removal
       ↓
Encoding
       ↓
Feature Scaling
       ↓
Outlier Detection
       ↓
Clean Dataset
```

---

# 2️⃣ Missing Values

Missing values are empty or unavailable values in a dataset.

### Check Missing Values

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.isnull().sum())
```

### Fill Numerical Missing Values

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
```

### Fill Using Median

```python
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
```

### Fill Categorical Values

```python
df["City"] = df["City"].fillna(df["City"].mode()[0])
```

---

# 3️⃣ Duplicate Values

Check duplicates:

```python
print(df.duplicated().sum())
```

Remove duplicates:

```python
df = df.drop_duplicates()
```

Check dataset shape:

```python
print(df.shape)
```

---

# 4️⃣ Encoding

Machine Learning algorithms generally require numerical input.

Suppose:

```text
Gender
------
Male
Female
Male
Female
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])
```

Example:

```text
Female → 0
Male   → 1
```

---

# 5️⃣ One-Hot Encoding

One-Hot Encoding converts categorical values into separate binary columns.

```python
df = pd.get_dummies(
    df,
    columns=["City"],
    drop_first=True
)
```

Example:

```text
City
----
Delhi
Noida
Delhi
```

can become:

```text
City_Noida
0
1
0
```

---

# 6️⃣ Feature Scaling

Feature scaling makes numerical features comparable in scale.

Two common methods:

```text
Standardization
Min-Max Normalization
```

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Formula:

```text
z = (x - mean) / standard deviation
```

---

# 7️⃣ Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

Formula:

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

Usually values are transformed into the range:

```text
0 to 1
```

---

# 🚨 8️⃣ Outlier Detection Using IQR

An outlier is an observation that is unusually far from the majority of the data.

Example:

```text
10
12
15
14
13
16
18
200
```

Here, `200` may be an outlier.

---

## What is IQR?

IQR stands for:

> Interquartile Range

It represents the range between the first quartile and third quartile.

```text
IQR = Q3 - Q1
```

Where:

```text
Q1 = 25th percentile
Q3 = 75th percentile
```

### Outlier Boundaries

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Any value:

```text
value < Lower Bound
```

or

```text
value > Upper Bound
```

can be considered an outlier under the IQR rule.

---

# 🧪 Complete IQR Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Select numerical column
column = "Salary"

# Calculate Q1 and Q3
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Detect outliers
outliers = df[
    (df[column] < lower_bound) |
    (df[column] > upper_bound)
]

print("Number of Outliers:", len(outliers))
print(outliers)
```

---

# 🧹 Remove Outliers

```python
df_clean = df[
    (df[column] >= lower_bound) &
    (df[column] <= upper_bound)
]

print("Original Shape:", df.shape)
print("Clean Shape:", df_clean.shape)
```

---

# 📊 Visualize Outliers

### Before Removing Outliers

```python
plt.figure(figsize=(8, 5))

sns.boxplot(y=df[column])

plt.title("Before Outlier Removal")
plt.show()
```

### After Removing Outliers

```python
plt.figure(figsize=(8, 5))

sns.boxplot(y=df_clean[column])

plt.title("After Outlier Removal")
plt.show()
```

---

# 📦 Boxplot

A boxplot is commonly used to visualize the distribution of numerical data and identify potential outliers.

```text
        |
        |       Upper Whisker
        |
    ┌─────────┐
    │         │
    │   Box   │
    │         │
    └─────────┘
        |
        |
        •  Potential Outlier
        •  Potential Outlier
```

---

# ⚠️ Important Note About Outliers

Outliers should not automatically be removed.

An outlier can represent:

```text
Data Entry Error
Measurement Error
Rare Event
Genuine Extreme Value
Fraud
Important Business Case
```

Therefore, outliers should be investigated before removing them.

---

# 9️⃣ Train-Test Split

Train-test split divides the dataset into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

Here:

```text
80% → Training Data
20% → Testing Data
```

---

# 🔟 Classification

Classification is a supervised Machine Learning technique used to predict categorical classes.

Examples:

```text
Spam / Not Spam
Yes / No
Pass / Fail
Disease / No Disease
Customer Churn / No Churn
```

Common classification algorithms:

```text
Logistic Regression
KNN
Decision Tree
Random Forest
SVM
Naive Bayes
Voting Classifier
```

---

# 🗳️ 1️⃣1️⃣ Max Voting / Voting Classifier

Max Voting is an Ensemble Learning technique.

Instead of depending on a single model, multiple classification models are combined.

Example:

```text
                 ┌── Logistic Regression → Class A
Input Data ──────┼── Decision Tree       → Class B
                 └── KNN                 → Class A

Votes:

Class A → 2
Class B → 1

Final Prediction → Class A
```

The class receiving the highest number of votes becomes the final prediction in **Hard Voting**.

---

# 🧠 Why Use Voting?

Different Machine Learning algorithms can learn different patterns from the same dataset.

For example:

```text
Logistic Regression
        +
Decision Tree
        +
KNN
        ↓
Voting Classifier
        ↓
Final Prediction
```

Combining models can provide a more robust prediction than relying on one classifier alone, depending on the dataset and model configuration.

---

# 🔨 Hard Voting

Hard Voting uses the predicted class labels from each classifier.

Example:

```text
Model 1 → Class 1
Model 2 → Class 2
Model 3 → Class 1

Class 1 → 2 Votes
Class 2 → 1 Vote

Final → Class 1
```

---

# 💻 Complete Hard Voting Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create individual models
model1 = LogisticRegression(max_iter=1000)

model2 = DecisionTreeClassifier(
    random_state=42
)

model3 = KNeighborsClassifier(
    n_neighbors=5
)

# Create Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ("logistic_regression", model1),
        ("decision_tree", model2),
        ("knn", model3)
    ],
    voting="hard"
)

# Train model
voting_model.fit(X_train, y_train)

# Prediction
y_pred = voting_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Voting Classifier Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

# 🟢 Soft Voting

Soft Voting uses the predicted probabilities from individual classifiers.

For example:

```text
Model 1 → [0.10, 0.80, 0.10]
Model 2 → [0.20, 0.70, 0.10]
Model 3 → [0.15, 0.75, 0.10]
```

The probabilities are combined and the class with the highest combined probability is selected.

---

# 💻 Soft Voting Code

```python
voting_soft = VotingClassifier(
    estimators=[
        ("logistic_regression", model1),
        ("decision_tree", model2),
        ("knn", model3)
    ],
    voting="soft"
)

voting_soft.fit(X_train, y_train)

y_pred_soft = voting_soft.predict(X_test)

accuracy_soft = accuracy_score(
    y_test,
    y_pred_soft
)

print(
    "Soft Voting Accuracy:",
    accuracy_soft
)
```

---

# 🔍 Hard Voting vs Soft Voting

| Feature              | Hard Voting                    | Soft Voting                              |
| -------------------- | ------------------------------ | ---------------------------------------- |
| Uses                 | Class labels                   | Probabilities                            |
| Decision             | Majority vote                  | Probability aggregation                  |
| `voting`             | `"hard"`                       | `"soft"`                                 |
| Probability required | No                             | Yes                                      |
| Example              | 2 out of 3 models vote Class A | Class A has highest combined probability |

---

# 📈 Individual Models vs Voting

We can compare individual models with the Voting Classifier.

```python
models = {
    "Logistic Regression": model1,
    "Decision Tree": model2,
    "KNN": model3
}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    score = accuracy_score(
        y_test,
        prediction
    )

    print(name, ":", score)
```

Then evaluate the Voting Classifier:

```python
voting_model.fit(X_train, y_train)

voting_prediction = voting_model.predict(X_test)

voting_score = accuracy_score(
    y_test,
    voting_prediction
)

print("Voting Classifier:", voting_score)
```

---

# 🌳 1️⃣2️⃣ Decision Tree

Decision Tree is a supervised learning algorithm that makes decisions using a tree-like structure.

```text
             Age > 30?
              /      \
            Yes       No
            /          \
       Salary > 50K    Class B
         /     \
      Yes       No
      /          \
  Class A       Class B
```

Python:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 🌲 1️⃣3️⃣ Random Forest

Random Forest combines multiple Decision Trees.

```text
             Random Forest
                   |
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Tree 1       Tree 2      Tree 3
       ↓           ↓           ↓
     Class A     Class B     Class A
       └───────────┼───────────┘
                   ↓
              Final Class
```

Python:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
```

---

# ⚡ 1️⃣4️⃣ Boosting

Boosting is an ensemble technique where models are trained sequentially, with later models focusing more on difficult observations.

Popular boosting algorithms:

```text
AdaBoost
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

---

# 🎯 1️⃣5️⃣ Model Evaluation

After training a Machine Learning model, we need to evaluate its performance.

Common metrics:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC
```

---

# Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

Formula:

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

# Precision

Precision answers:

> Out of all observations predicted as positive, how many were actually positive?

```text
Precision =
TP / (TP + FP)
```

---

# Recall

Recall answers:

> Out of all actual positive observations, how many were correctly identified?

```text
Recall =
TP / (TP + FN)
```

---

# F1 Score

F1 Score combines Precision and Recall.

```text
F1 =
2 × (Precision × Recall)
------------------------
  Precision + Recall
```

Python:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 🔲 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

A binary confusion matrix contains:

```text
                 Predicted
                 Positive Negative

Actual Positive    TP        FN
Actual Negative    FP        TN
```

---

# 🔄 Cross Validation

Cross-validation evaluates a model across multiple train-validation splits.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    voting_model,
    X,
    y,
    cv=5
)

print("Scores:", scores)
print("Mean Score:", scores.mean())
```

---

# ⚙️ Hyperparameter Tuning

Hyperparameters are settings that are defined before model training.

Examples:

```text
n_estimators
max_depth
learning_rate
n_neighbors
C
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("Best Score:")
print(grid_search.best_score_)
```

---

# 🧪 Feature Engineering

Feature engineering means creating, transforming, or selecting features to improve the usefulness of input data for Machine Learning.

Examples:

```text
Feature Creation
Feature Transformation
Feature Selection
Feature Extraction
Encoding
Scaling
```

Example:

```python
df["BMI"] = df["Weight"] / (
    df["Height"] ** 2
)
```

---

# 📉 Feature Selection

Feature selection identifies the most useful features for a Machine Learning model.

Example:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(
    score_func=f_classif,
    k=3
)

X_selected = selector.fit_transform(
    X,
    y
)

print(X_selected.shape)
```

---

# 🧩 Complete Machine Learning Pipeline

A practical Machine Learning project can follow this structure:

```text
                Dataset
                   ↓
          Data Understanding
                   ↓
          Data Cleaning
                   ↓
         Missing Value Handling
                   ↓
             Duplicates
                   ↓
              Encoding
                   ↓
             Outliers
                   ↓
               Scaling
                   ↓
                 EDA
                   ↓
         Feature Engineering
                   ↓
          Train-Test Split
                   ↓
          Model Selection
                   ↓
          Model Training
                   ↓
             Prediction
                   ↓
           Model Evaluation
                   ↓
        Hyperparameter Tuning
                   ↓
          Final Model
```

---

# 📚 Learning Roadmap

## Beginner

* Python Basics
* NumPy
* Pandas
* Data Cleaning
* Matplotlib
* Seaborn
* Statistics
* EDA

## Intermediate

* Train-Test Split
* Feature Engineering
* Encoding
* Scaling
* Outlier Detection
* Linear Regression
* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* SVM
* Naive Bayes

## Advanced

* Ensemble Learning
* Bagging
* Boosting
* Voting
* Stacking
* Cross Validation
* Hyperparameter Tuning
* Feature Selection
* Dimensionality Reduction
* XGBoost
* Model Pipelines

---

# 🎯 Current Practice Topics

```text
✅ Data Preprocessing
✅ Missing Value Handling
✅ Duplicate Removal
✅ Encoding
✅ Feature Scaling
✅ IQR Outlier Detection
✅ IQR Outlier Removal
✅ Train-Test Split
✅ Classification
✅ Logistic Regression
✅ KNN
✅ Decision Tree
✅ Random Forest
✅ Max Voting
✅ Hard Voting
✅ Soft Voting
✅ Model Evaluation
✅ Cross Validation
✅ Hyperparameter Tuning
```

---

# 💡 Key Concepts

### IQR

```text
IQR = Q3 - Q1
```

Used for detecting potential outliers.

### Max Voting

```text
Multiple Models
      ↓
Individual Predictions
      ↓
Majority Vote
      ↓
Final Prediction
```

### Hard Voting

```text
Predicted Classes
      ↓
Majority Vote
      ↓
Final Class
```

### Soft Voting

```text
Predicted Probabilities
      ↓
Combine Probabilities
      ↓
Highest Probability
      ↓
Final Class
```

---

# 🏆 Learning Objective

The main objective of this repository is to develop a strong practical understanding of Machine Learning by implementing concepts through Python and Scikit-learn.

This repository will continuously be updated with new:

* Algorithms
* Techniques
* Datasets
* Experiments
* Visualizations
* Machine Learning projects

---

# 👨‍💻 Author

**Prince Kumar**

B.Tech – Computer Science & Engineering
Artificial Intelligence & Machine Learning

### Profiles

* LinkedIn: https://www.linkedin.com/in/prince-kumar-125396321/
* GitHub: https://github.com/princekumarwcm-max

---

# ⭐ Repository Goal

> Learn Machine Learning by implementing concepts, experimenting with data, and building practical solutions.

If you find this repository useful, consider giving it a ⭐.
