# Importing Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Load Dataset
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

# Splitting Features and Target
X = data.drop("Class", axis=1)  # Features
y = data["Class"]              # Target column

# 1. Balance Dataset
# Using SMOTE to balance the dataset
smote = SMOTE(random_state=11)
X_balanced, y_balanced = smote.fit_resample(X, y)

# 2. Sampling Methods
# Function to validate that the sample contains at least 2 classes
def validate_sample(X_sample, y_sample):
    if len(y_sample.unique()) < 2:
        raise ValueError("Sample contains only one class. Adjust the sampling method or parameters.")

# Simple Random Sampling
def simple_random_sampling(X, y, n=500):
    idx = np.random.choice(X.index, size=n, replace=False)
    X_sample, y_sample = X.loc[idx], y.loc[idx]
    validate_sample(X_sample, y_sample)
    return X_sample, y_sample

# Systematic Sampling
def systematic_sampling(X, y, step=5):
    idx = np.arange(0, len(X), step)
    X_sample, y_sample = X.iloc[idx], y.iloc[idx]
    validate_sample(X_sample, y_sample)
    return X_sample, y_sample

# Stratified Sampling
def stratified_sampling(X, y, n=500):
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=42)
    validate_sample(X_sample, y_sample)
    return X_sample, y_sample

# Cluster Sampling
def cluster_sampling(X, y, n_clusters=5):
    clusters = np.array_split(X.index, n_clusters)
    selected_cluster = np.random.choice(range(len(clusters)), size=1, replace=False)
    selected_idx = clusters[selected_cluster[0]]
    X_sample, y_sample = X.loc[selected_idx], y.loc[selected_idx]
    validate_sample(X_sample, y_sample)
    return X_sample, y_sample

# Convenience Sampling
def convenience_sampling(X, y, n=500):
    X_sample, y_sample = X.iloc[:n], y.iloc[:n]
    validate_sample(X_sample, y_sample)
    return X_sample, y_sample

# Create Samples using the above techniques
samples = [
    simple_random_sampling(X_balanced, y_balanced, n=500),
    systematic_sampling(X_balanced, y_balanced, step=10),
    stratified_sampling(X_balanced, y_balanced, n=500),
    cluster_sampling(X_balanced, y_balanced, n_clusters=5),
    convenience_sampling(X_balanced, y_balanced, n=500),
]

# 3. Define ML Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=11),
    "RandomForest": RandomForestClassifier(random_state=11),
    "SVM": SVC(random_state=11),
    "DecisionTree": DecisionTreeClassifier(random_state=11),
    "KNN": KNeighborsClassifier()
}

# 4. Evaluate Sampling Techniques with ML Models
results = pd.DataFrame(index=models.keys(), columns=[f"Sampling{i+1}" for i in range(5)])

for model_name, model in models.items():
    for i, (X_sample, y_sample) in enumerate(samples):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=11)
        
        # Train and evaluate the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store accuracy
        results.loc[model_name, f"Sampling{i+1}"] = accuracy

# Print Results
print("Accuracy Table:")
print(results)

# Save Results to CSV
results.to_csv("accuracy_results.csv", index=True)
