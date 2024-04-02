# Wine Club Subscription Prediction Pipeline

This repository contains a machine learning pipeline to predict whether customers will subscribe to a wine club based on their first order.

## Command to Run

```bash
python3 main.py --tuning_train_size 0.01 --models random_forest svm xgboost --param_grids "{'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15]}}" "{'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}}" "{'xgboost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}}"
```

| Parameter                 | Description                                                                                                                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--tuning_train_size`     | The size of the training set to be used for hyperparameter tuning. In this example, it's set to 0.01, indicating 1% of the training data.                   |
| `--models`                | List of machine learning models to be trained and evaluated. In this example, it includes Random Forest, Support Vector Machine (SVM), and XGBoost.          |
| `--param_grids`           | Hyperparameter grids for each model specified in the `--models` parameter. The hyperparameters specified here will be tuned using GridSearchCV. For example: |
|                           | - For Random Forest: `{'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15]}`                                                                           |
|                           | - For SVM: `{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}`                                                                                              |
|                           | - For XGBoost: `{'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}`                                                                                  |
## Pipeline operations
| Preprocessing            | Tuning                                        | Training                               | Evaluation                               |
|--------------------------|-----------------------------------------------|----------------------------------------|------------------------------------------|
| New features creation   | GridSearch with cross-validation (find best model-hyperparameters) | Use all training set to train the best model | Evaluate the trained model on test set |
| Train-test splitting    |                                               |                                        |                                          |
| Label encoding          | Size of cross validation (K-fold)             |                                        |                                          |
| Normalization           |                                               |                                        |                                          |
| Resampling              |                                               |                                        |                                          |
| Missing values treatment|                                               |                                        |                                          |

## Exploratory Data Analysis

An exploratory data analysis was performed (notebooks/eda.ipynb) in order to gather the following information:

#### Are there any missing values in the dataset, and if so, how should they be handled?
NaN values on 'intro_tier_1' --> could be treated as a new label

#### Are there any outliers or anomalies in the data that need to be addressed?
Insignificant amount of negative values, and does not seem to have outliers

#### What is the proportion of users that became full_club?
7.68%

#### How do the distributions of features differ between full club members and non-members?
It does not seem to have significative differences

#### Which month of the year are users who just sign up most likely to become full_club?
Users who bought on August/September seems to be more likely to become full_club. Probably due to the beginning of Fall and the Winter after that, with holidays such as Thanksgiving, Christimas and New Year's Eve, and also a a colder weather that may encourage people to stay indoors and enjoy a glass of wine. So they probably antecipated that and took advantage of the wine subscription.

However, users who bought on the November and December probably only bought some bottles of wine for the holidays but they are not a wine connoisseur.

#### Which day of the week are users who just sign up most likely to become full_club?
Users who make purchases on weekends appear to have a slightly increased likelihood of becoming full_club. However the disparity seems negligible

#### Can new features be derived from existing ones to improve predictive performance?
Days to delivery may be a good new feature

#### What is the relationship between intro_tier and users becoming full_club?
Users who found Firstleaf through Search and Customer referral tends to have a higher probability of becoming full club 

## Configuration Environment

| Variable      | Description                                    | Example                        |
|---------------|------------------------------------------------|--------------------------------|
| `DATASET_PATH`| Path to the dataset file                       | "data/available_intro_user_data.csv" |
| `RANDOM_STATE`| Seed for random number generation              | 0                              |
| `CROSS_VALIDATION_SIZE`| Size of cross validation (K-fold)              | 5                              |
| `TUNING_SCORE_METRIC`| Metric to score model selection and hyperparameter tuning              | "f1"                              |
| `RESAMPLING_STRATEGY`| Resampling strategy for training set (RandomOverSampler, SMOTE, ADASYN, RandomUnderSampler, NearMiss, None) | "SMOTE"                              |

## Project Structure
```bash
.
├── config
│   ├── config.env              # Configuration file with environment variables
│   └── config.py               # Configuration script
├── data
│   └── available_intro_user_data.csv    # Dataset file
├── mlruns                       # MLflow experiment informations, logs, and trained models
├── models
│   ├── last_trained_model_results.json   # Test metrics of last trained model
│   └── last_trained_model.joblib         # Last trained model file
├── notebooks
│   └── eda.ipynb              # Exploratory data analysis
├── src
│   ├── data_preprocessing.py   # Encoding, normalization, null values, and other preprocessing operations
│   ├── data_splitting.py       # Separate dataset into train and test sets
│   ├── evaluating.py           # Test metrics to evaluate the trained model
│   ├── feature_engineering.py  # Creation of new features to improve model performance
│   ├── load_dataset.py         # Loading and preparation of dataset
│   ├── model.py                # Model instantiation
│   ├── training.py             # Training script
│   └── tuning.py               # Find the best model and hyperparameters based on GridSearch with Cross Validation
├── utils
│   └── create_experiment_name.py    # Automatically create name of experiments based on datetime
└── main.py                     # Main script
```