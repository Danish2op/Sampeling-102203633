# Sampling Techniques and Machine Learning Models for Credit Card Fraud Detection

This project demonstrates the application of **five different sampling techniques** to balance a highly imbalanced dataset and evaluate the performance of **five machine learning models** on the dataset. The goal is to identify the best combination of sampling technique and model for detecting fraudulent transactions in credit card data.

## Project Overview

In this project, we use the `Creditcard_data.csv` dataset, which consists of credit card transaction data. The dataset has 31 columns, including the `Class` column, which indicates whether a transaction is fraudulent (`1`) or not (`0`). We perform the following tasks:

1. **Balance the Dataset** using oversampling (SMOTE).
2. **Apply Five Different Sampling Techniques** to create five different datasets.
3. **Train and Evaluate Five Machine Learning Models** on each of the sampled datasets.
4. **Compare the Results** of each combination of sampling technique and model to determine the most effective approach.

## Sampling Techniques Used

1. **Simple Random Sampling**: Randomly selects a subset of data without considering any distribution or stratification.
2. **Systematic Sampling**: Selects every `k`-th row from the dataset, ensuring an even spread.
3. **Stratified Sampling**: Divides the dataset into strata (subgroups) based on the target variable (`Class`), ensuring that each class is proportionally represented.
4. **Cluster Sampling**: Divides the data into clusters, and samples entire clusters.
5. **Convenience Sampling**: Selects the first `n` rows from the dataset.

## Machine Learning Models Used

The following machine learning models were evaluated using the sampled data:

1. **Logistic Regression**: A statistical method for binary classification.
2. **Random Forest**: An ensemble method that combines multiple decision trees for better generalization.
3. **Support Vector Machine (SVM)**: A classifier that finds the hyperplane separating classes.
4. **Decision Tree**: A model that splits data into nodes based on feature values to make predictions.
5. **K-Nearest Neighbors (KNN)**: A non-parametric classifier that classifies based on the closest data points.

## Methodology

1. **Balancing the Dataset**: The dataset is highly imbalanced, with the majority of transactions being non-fraudulent. To handle this, we use **SMOTE (Synthetic Minority Over-sampling Technique)**, which generates synthetic samples for the minority class.
  
2. **Sampling**: The dataset is divided into five different subsets using the five sampling techniques mentioned above.

3. **Model Training and Evaluation**: For each sampling technique, five machine learning models are trained and evaluated on the dataset. The evaluation is done using **accuracy** as the primary performance metric.

4. **Results**: The performance of each model is compared across the different sampling techniques.

## Results

Here is a summary of the accuracy obtained by each model using different sampling techniques:

| Model               | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|---------------------|-----------|-----------|-----------|-----------|-----------|
| **LogisticRegression** | 0.89      | 0.91      | 0.93      | 0.93      | 0.99      |
| **RandomForest**      | 0.99      | 1.0       | 0.99      | 0.99      | 1.0       |
| **SVM**               | 0.69      | 0.65      | 0.62      | 0.79      | 1.0       |
| **DecisionTree**      | 0.97      | 0.89      | 0.99      | 0.85      | 1.0       |
| **KNN**               | 0.77      | 0.70      | 0.72      | 0.81      | 1.0       |

### **Key Observations:**
- **RandomForest** and **Logistic Regression** consistently performed well across most sampling techniques.
- **Support Vector Machine (SVM)** performed poorly for most sampling techniques but achieved perfect accuracy with **Sampling5**.
- **Convenience Sampling** and **Stratified Sampling** showed a good balance in performance for all models.
  
The **best performing combination** was the **RandomForest model** combined with **Sampling5**, achieving a perfect accuracy score of **1.0**.

## Instructions for Running the Code

To run the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourGitHubUsername/Sampling_Assignment.git
   cd Sampling_Assignment
#   1 0 2 2 0 3 6 3 3 - S a m p e l i n g  
 