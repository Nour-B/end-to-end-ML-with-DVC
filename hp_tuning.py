import json

import pandas as pd 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from utils_and_constants import PROCESSED_DATASET, get_hp_tuning_results, load_data


def main():
    # Load the data
    X,y = load_data(PROCESSED_DATASET)
    print(X)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=42)

    # Define the model and hyperparameter serach space
    model = LogisticRegression()
    params_grid = json.load(open("hp_config.json","r"))

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=5, n_jobs= -1, verbose=2)
    grid_search.fit(X_train,y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # print and save the best hyperparameters
    print("====================Best Hyperparameters==================")
    print(json.dumps(best_params, indent=2))
    print("==========================================================")

    with open("logreg_best_params.json", "w") as outfile:
        json.dump(best_params, outfile)

    # Print and save the results of hyperparameter tuning
    markdown_table = get_hp_tuning_results(grid_search)
    with open("hp_tuning_results.md", "w") as markdown_file:
        markdown_file.write(markdown_table)


if __name__ == "__main__":
    main()
