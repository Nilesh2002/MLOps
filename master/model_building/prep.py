# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Import for handling missing values

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOK"))
#DATASET_PATH = "hf://datasets/nilku/Tourism-Packag-Prediction/tourism.csv"
DATASET_PATH = "hf://datasets/nilku/TourismPackagePrediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                   # Age of the customer
    'DurationOfPitch',       # Duration of the sales pitch
    'NumberOfPersonVisiting',# Number of persons visiting with the customer
    'NumberOfFollowups',     # Number of follow-ups done by the sales team
    'NumberOfTrips',         # Number of trips taken by the customer
    'MonthlyIncome',         # Customer's monthly income
    'NumberOfChildrenVisiting' # Number of children visiting with the customer
  ]

# List of categorical features in the dataset
categorical_features = [
    'Gender',                # Gender of the customer
    'MaritalStatus',         # Marital status of the customer
    'TypeofContact',         # Type of contact for the customer
    'Occupation',            # Occupation of the customer
    'ProductPitched',        # Product pitched to the customer
    'Designation',           # Designation
    'CityTier',              # Type of dwelling
    'PreferredPropertyStar', # Preferred hotel property star rating (moved from numeric)
    'Passport',              # Whether the customer has a passport (0=No, 1=Yes) (moved from numeric)
    'PitchSatisfactionScore',# Customer's satisfaction score with the pitch (moved from numeric)
    'OwnCar'                 # Whether the customer owns a car (0=No, 1=Yes) (moved from numeric)
]

# Define predictor matrix (X) using selected numeric and categorical features
# Drop ID column from data cleaning steps.
X = tourism_dataset[numeric_features + categorical_features].copy()

# Define target variable
y = tourism_dataset[target].copy()

# --- Data Cleaning Steps ---

# 1. Handle missing values in numerical features with median imputation
if numeric_features:
    numeric_imputer = SimpleImputer(strategy='median')
    X.loc[:, numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    print(f"Imputed missing numerical values with median for features: {numeric_features}")

# 2. Handle missing values in categorical features with mode imputation
if categorical_features:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X.loc[:, categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    print(f"Imputed missing categorical values with mode for features: {categorical_features}")

# 3. Handle missing values in target variable by dropping corresponding rows
missing_target_rows = y.isnull()
if missing_target_rows.any():
    print(f"Dropping {missing_target_rows.sum()} rows with missing target values.")
    X = X[~missing_target_rows].reset_index(drop=True)
    y = y[~missing_target_rows].reset_index(drop=True)
    print(f"Shape of X after dropping missing target rows: {X.shape}")
    print(f"Shape of y after dropping missing target rows: {y.shape}")

# Ensure that the index is reset after dropping rows, to avoid issues with train_test_split
# X = X.reset_index(drop=True)
# y = y.reset_index(drop=True)

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="nilku/TourismPackagePrediction",
        repo_type="dataset",
        commit_message=f"Update {file_path.split('/')[-1]} after data cleaning"
    )
