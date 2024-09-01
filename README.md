# Bank Customer Churn Prediction Model

## Introduction
The **Bank Customer Churn Prediction Model** project focuses on predicting whether customers are likely to churn (leave the bank) based on various attributes related to their accounts and demographics. This model helps identify at-risk customers, allowing the bank to take proactive steps to retain them. By leveraging machine learning techniques, particularly Support Vector Machines (SVM), and addressing data imbalance through undersampling and oversampling methods, the project aims to build a robust predictive model that can enhance customer retention strategies.

## Objective
The primary objective of this project is to develop a machine learning model that accurately predicts customer churn. This model will provide insights into the factors contributing to customer attrition, enabling the bank to implement targeted retention strategies and improve overall customer satisfaction.

## Project Outline

### 1. Import Library
This step involves importing the necessary Python libraries for data analysis, visualization, and modeling:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn

### 2. Import Data URL
The dataset used for this project is publicly available and can be accessed at the following URL:
[Bank Churn Modelling Dataset](https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv).

### 3. Analyze Data
In this step, the dataset is explored to understand its structure and content. Key actions include checking for missing values, identifying duplicate entries, and generating basic statistics to understand the distribution of data.

### 4. Data Encoding
To prepare the data for modeling, categorical variables such as 'Gender' and 'Geography' are encoded into numerical formats using mapping techniques. This step is crucial as machine learning models require numerical input.

### 5. Define Label and Features
The target variable (label) for prediction is **Churn**, which indicates whether a customer has churned or not. The remaining features (independent variables) are used to predict this outcome.

### 6. Handling Imbalance Data
Given the imbalance in the dataset, where the number of non-churn cases significantly outweighs churn cases, techniques like undersampling and oversampling are applied. These techniques help balance the dataset, leading to more reliable and accurate model predictions.

### 7. Undersampling and Oversampling
- **Random Under Sampling:** Reduces the number of non-churn cases to match the number of churn cases, balancing the dataset by decreasing the majority class.
- **Random Over Sampling:** Increases the number of churn cases to match the number of non-churn cases, balancing the dataset by boosting the minority class.

### 8. Train Test Split Dataset
The data is split into training and testing sets to evaluate the model's performance. Typically, 70% of the data is used for training, and 30% is reserved for testing.

### 9. Standardize Features
The numerical features are standardized to have a mean of 0 and a standard deviation of 1. This step is essential for improving the performance of certain machine learning models, including SVM.

### 10. Support Vector Machine Classifier with Raw Data
An initial Support Vector Machine (SVM) classifier is trained on the raw, unbalanced dataset to establish a baseline model. This model serves as a reference point for further improvements.

### 11. Model Accuracy
The accuracy of the baseline model is evaluated using the testing set. Performance metrics such as precision, recall, and F1-score are calculated to assess the model's effectiveness.

### 12. Hyperparameter Tuning
GridSearchCV is used to fine-tune the SVM model's hyperparameters, including `C`, `gamma`, and `kernel`. This step aims to optimize the model's performance by finding the best combination of parameters.

### 13. Model with Random Under Sampling
The SVM model is retrained using the undersampled dataset. The model's performance is evaluated to determine if the undersampling technique has improved prediction accuracy.

### 14. Model Accuracy
The accuracy and performance metrics of the undersampled model are assessed and compared with the baseline model to evaluate the effectiveness of undersampling.

### 15. Hyperparameter Tuning
Further hyperparameter tuning is conducted on the undersampled dataset to refine the model and enhance its predictive capabilities.

### 16. Model with Random Over Sampling
The SVM model is retrained using the oversampled dataset. The model's performance is evaluated to determine if oversampling has improved prediction accuracy.

### 17. Model Accuracy
The accuracy and key metrics of the oversampled model are assessed and compared with both the baseline and undersampled models to determine the best approach.

### 18. Hyperparameter Tuning
Hyperparameter tuning is performed on the oversampled dataset to further optimize the model for the best possible predictions.

## Dataset
The dataset used in this project is available at the following link:
[Bank Churn Modelling Dataset](https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv).

## Conclusion
This project demonstrates the process of building a machine learning model to predict bank customer churn. By addressing data imbalance and optimizing model parameters, the project highlights the importance of careful data preparation and model tuning in achieving accurate predictions. The final model can be used by banks to identify at-risk customers and implement strategies to improve customer retention.

