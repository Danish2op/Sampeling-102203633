
---

## 📊 **Dataset Information**

The dataset consists of **772 rows** and **31 columns** with the following attributes:
- **Features** (`V1` to `V28`, `Time`, `Amount`): These represent anonymized features and the transaction amount/time.
- **Target** (`Class`): A binary classification target indicating whether a transaction is fraudulent (1) or not (0).

---

## 🚀 **Methodology**

### **1. Data Preprocessing and Balancing**
- **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the dataset by increasing the number of instances in the minority class (fraudulent transactions).
- This helps address the class imbalance, ensuring better model performance for detecting fraud.

### **2. Sampling Techniques**
Five different sampling techniques were implemented to create subsets of the dataset:

1. **Simple Random Sampling**: Randomly selects a sample of data points.
2. **Systematic Sampling**: Selects every `k`-th row from the dataset.
3. **Stratified Sampling**: Ensures that each class (fraudulent/non-fraudulent) is represented proportionally.
4. **Cluster Sampling**: Divides the dataset into clusters and randomly selects one or more clusters to represent the entire population.
5. **Convenience Sampling**: Selects the first `n` rows of the dataset.

### **3. Machine Learning Models**
Five different machine learning models were used to evaluate the performance of each sampling technique:
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**

### **4. Evaluation Metrics**
The models' performance was evaluated using **accuracy** as the primary metric. The accuracy was computed for each model across the five sampling techniques.

---

## 📈 **Results**

The accuracy results for each machine learning model across the five sampling techniques are summarized in the table below:

| Model               | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|---------------------|-----------|-----------|-----------|-----------|-----------|
| **Logistic Regression**  | 0.89      | 0.91      | 0.93      | 0.93      | **0.99**  |
| **Random Forest**        | 0.99      | **1.00**  | 0.99      | 0.99      | **1.00**  |
| **SVM**                 | 0.69      | 0.65      | 0.62      | 0.79      | **1.00**  |
| **Decision Tree**        | 0.97      | 0.89      | 0.99      | 0.85      | **1.00**  |
| **KNN**                 | 0.77      | 0.70      | 0.72      | 0.82      | **1.00**  |

**Key Insights**:
- **Random Forest** and **SVM** performed the best across all sampling techniques, achieving **perfect accuracy** (1.00) in most cases.
- **Logistic Regression** showed a significant improvement when using **Sampling5** (Convenience Sampling).
- **KNN** and **Decision Tree** also performed well, with some models achieving **1.00 accuracy** in specific sampling techniques.

