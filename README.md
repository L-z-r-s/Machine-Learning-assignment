# Heart Disease Prediction Using Machine Learning

## 1. Problem Statement

Heart disease is one of the major causes of mortality worldwide. Early identification of individuals who may be at risk of heart disease can help support timely medical intervention and better healthcare decisions.

The objective of this project is to develop a machine learning-based classification system that predicts the presence or absence of heart disease using patient medical attributes. Five different machine learning classification algorithms are implemented, evaluated, and compared to identify the model that provides the best predictive performance.

The project also applies hyperparameter tuning using GridSearchCV to optimize model performance and improve classification results.

## 2. Dataset Description

The project uses a Heart Disease dataset containing 1,025 patient records and 13 input features.

**Dataset characteristics:**
- Total instances: 1,025
- Input features: 13
- Target variable: `target`
- Problem type: Binary Classification
- Number of classes: 2

**Input Features:**

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Gender indicator |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia-related category |

**Target Variable:**

The `target` column represents the binary classification output, with two classes: 0 and 1.

The dataset contains 526 records belonging to class 1 and 499 records belonging to class 0.

## 3. GitHub Repository Link

**GitHub Repository:** [https://github.com/L-z-r-s/Machine-Learning-assignment]

The repository contains the machine learning implementation, dataset used for testing, trained model files, preprocessing artifacts, and application source code.

## 4. Models Used and Evaluation

Five machine learning classification algorithms were implemented using Scikit-learn.

### 4.1 Machine Learning Algorithms

**1. Logistic Regression**

Logistic Regression is a supervised classification algorithm that estimates the probability of a data point belonging to a particular class using the logistic function. It is used as a baseline classification model.

**2. Decision Tree Classifier**

Decision Tree is a supervised learning algorithm that makes predictions using a tree-like structure of decision rules. It recursively divides the dataset based on feature conditions to classify patient records.

**3. K-Nearest Neighbors (KNN)**

KNN is a distance-based classification algorithm that predicts the class of a new observation based on the majority class among its nearest neighbors. The implementation uses Euclidean distance.

**4. Naive Bayes Classifier**

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes conditional independence between input features and models continuous features using Gaussian distributions.

**5. Support Vector Machine (SVM)**

SVM is a supervised classification algorithm that identifies an optimal decision boundary to separate different classes. The implementation uses the Radial Basis Function (RBF) kernel.

### 4.2 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Heart Disease dataset using Pandas.
2. Separated the independent variables (13 features) and dependent variable (`target`).
3. Divided the dataset into training and testing sets using an 80:20 split.
4. Used stratified sampling to maintain the class distribution.
5. Applied StandardScaler to normalize the feature values.
6. Saved the test dataset as `test_data.csv`.
7. Saved the scaler and trained classification models using Joblib.

The dataset was divided using `random_state=42` to maintain reproducibility.

### 4.3 Hyperparameter Optimization

GridSearchCV was used to identify suitable hyperparameters for all five classification algorithms.

The optimization process used 5-fold cross-validation with F1-score as the scoring metric.

| Model | Best Hyperparameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=200, solver=lbfgs |
| Decision Tree | criterion=entropy, max_depth=10, min_samples_split=2 |
| KNN | metric=euclidean, n_neighbors=9, weights=distance |
| Naive Bayes | var_smoothing=1.0 |
| SVM | C=10, gamma=auto, kernel=rbf |

### 4.4 Model Evaluation Metrics

The performance of each classification model was evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC Score:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures the proportion of positive predictions that are actually positive.
- **Recall:** Measures the proportion of actual positive observations correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the overall quality of binary classification using all four confusion matrix outcomes.

### 4.5 Model Comparison Table

The following table presents the evaluation results of the fine-tuned classification models.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.8244 | 0.9105 | 0.7634 | 0.9524 | 0.8475 | 0.6686 |
| Support Vector Machine | 0.9854 | 1.0000 | 0.9722 | 1.0000 | 0.9859 | 0.9711 |

### 4.6 Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an accuracy of 80.98% and an AUC score of 92.98%. It demonstrated good recall of 91.43%, indicating that it identified a large proportion of positive cases. However, its overall accuracy and MCC were lower than those of the other models.

**Decision Tree**

The fine-tuned Decision Tree achieved 100% across all evaluation metrics. It successfully classified every observation in the test dataset. This indicates excellent performance on the current test split, although further validation is required to establish its generalization ability.

**K-Nearest Neighbors**

KNN achieved 100% across all evaluation metrics after hyperparameter optimization. The distance-weighted KNN model with nine neighbors performed exceptionally well on the test dataset.

**Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 82.44% and an F1-score of 84.75%. Its recall of 95.24% was relatively high, suggesting that it identified most positive cases. However, its precision was comparatively lower.

**Support Vector Machine**

SVM achieved an accuracy of 98.54%, an AUC score of 1.0000, and an F1-score of 98.59%. It demonstrated an excellent balance between precision and recall and achieved an MCC score of 0.9711.

### 4.7 Overall Winner

Based on the evaluation results, **Decision Tree and KNN are the top-performing models**, both achieving 100% across all six evaluation metrics.

However, the Support Vector Machine is also a strong performer, achieving 98.54% accuracy and 97.11% MCC.

The perfect Decision Tree and KNN results should be independently validated, particularly by checking for duplicate or overlapping records and evaluating performance on an independent dataset. Therefore, the results should not automatically be interpreted as proof of perfect real-world prediction.

## 5. Project Structure

The recommended GitHub repository structure is:

```text
Heart-Disease-Prediction/
│
├── app.py
├── train_and_export.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── heart.csv
│
├── model/
│   ├── scaler.pkl
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── KNN.pkl
│   ├── Naive_Bayes.pkl
│   └── SVM.pkl

```

*Note: The structure above is the recommended repository layout. Ensure the files are actually added before publishing the repository.*

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

## 7. Installation and Execution

### Prerequisites

Install Python and the required dependencies.

### Step 1: Clone the Repository

```bash
git clone https://github.com/L-z-r-s/Machine-Learning-assignment.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Machine-Learning-assignment
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook test_and_export.ipynb
```

Execute the notebook cells to perform preprocessing, model training, hyperparameter optimization, and evaluation.

### Step 5: Run the Streamlit Application

Once `app.py` is implemented, launch the application using:

```bash
streamlit run app.py
```

## 8. Streamlit Application

The project assignment requires an interactive Streamlit application with the following features:

- CSV test dataset upload.
- Dropdown selection of the five trained classification models.
- Loading of saved trained models and preprocessing artifacts.
- Display of accuracy, AUC, precision, recall, F1-score, and MCC.
- Confusion matrix or classification report.
- Comparison of the performance of different classification models.

The application should evaluate uploaded test data using previously trained models without retraining on the uploaded data.

## 9. Deployment

The application is intended to be deployed using Streamlit Community Cloud.

**Live Streamlit Application:** [(https://ml2026aj05230.streamlit.app)]

**GitHub Repository:** [(https://github.com/L-z-r-s/Machine-Learning-assignment)]

## 10. Conclusion

This project implements and evaluates five machine learning classification algorithms for heart disease prediction using patient medical attributes.

The models were optimized using GridSearchCV and evaluated using six classification metrics. The experimental results indicate that Decision Tree and KNN achieved the highest scores, followed by Support Vector Machine.

The project demonstrates the application of machine learning techniques in healthcare-related classification problems and provides a foundation for an interactive prediction and model evaluation application.
