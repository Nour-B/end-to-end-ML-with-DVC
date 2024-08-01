import json

import pandas as pd
from sklearn.model_selection import GridSearchCV

RAW_DATASET = "raw_dataset/cc_approvals.data"
PROCESSED_DATASET = "processed_dataset/cc_approvals.csv"
TARGET_COLUMN = -1 # index of the last column in the dataframe which is the target variable
TOL = 0.0001
MAX_ITER=100



def load_data(file_path):
    data= pd.read_csv(file_path)
    X = data.iloc[:, :TARGET_COLUMN].values
    y= data.iloc[:, [TARGET_COLUMN]].values
    return X, y


def load_hyperparameters(hyperparameter_file):
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters



def get_hp_tuning_results(grid_search: GridSearchCV) -> str:
    """Get the results of hyperparameter tuning in a Markdown table"""
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Extract and split the 'params' column into subcolumns
    params_df = pd.json_normalize(cv_results["params"])

    # Concatenate the params_df with the original DataFrame
    cv_results = pd.concat([cv_results, params_df], axis=1)

    # Get the columns to display in the Markdown table
    cv_results = cv_results[
        ["rank_test_score", "mean_test_score", "std_test_score"]
        + list(params_df.columns)
    ]

    cv_results.sort_values(by="mean_test_score", ascending=False, inplace=True)
    return cv_results.to_markdown(index=False)