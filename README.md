# End To End ML 

## About

This is an end-to-end machine learning project that automates the credit card approval process for banks using supervised learning techniques. The idea of this project comes from the DataCamp platform [(link)](https://app.datacamp.com/learn/projects/1908).
The project includes data preprocessing, model training, and evaluation. It uses DVC for data versioning and GitHub Actions for continuous integration.


## Table of Contents

- [Data](#data)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [GitHub Actions Workflows](#github-actions-workflows)




## Data

The data is a small subset of the **Credit Card Approval** dataset from the UCI Machine Learning Repository showing the credit card applications a bank receives. The last column in the dataset is the target value.
It can be downloaded from [this link](https://archive.ics.uci.edu/dataset/27/credit+approval).

## Technologies Used

- Python 3.x
- Pandas 
- NumPy 
- Scikit-learn 
- Matplotlib 
- DVC 
- GitHubActions
- VS Code Dev Container


## Setup Instructions

### Prerequisites
1. Python 3.x
2. pip
> [!NOTE]  
> This repository includes a .devcontainer configuration for a reliable development environment using VS Code.



### Installation

1. Clone the repository:

```bash
git clone https://github.com/Nour-B/end-to-end-ML-with-DVC.git
cd end-to-end-ML-with-DVC
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize DVC

```bash
dvc init
```
4. Pull the data:

```bash
dvc pull
```

> [!IMPORTANT]  
> The `dvc.yaml` file includes the DVC pipeline orchestrating the lists of stages, commands, dependencies, and outputs used in this project. 

## Usage

### Data preprocessing 
After setting up the project, you can preprocess the data using the following script.

```bash
python preprocess_dataset.py
```

### Model Training and Evaluation

The model is trained and evaluated using the `train.py`. This script does the following:

1. Loads the data from processed_data.csv.
2. Trains a Logistic Regression model.
3. Evaluates the model performance using the following metrics: Accuracy, Precision, f1_score and recall
4. Saves the metrics to metrics.json.
5. Stores predictions data for the confusion matrix.


## GitHub Actions Workflows

The CI/CD pipelines are defined in `.github/workflows/train.yaml` and `.github/workflows/hp_cml.yaml`. 
- `.github/workflows/train.yaml` achieves the hyperparameter tuning and uses CML GitHub Action to open a new pull request from a new training branch to main that will run the training pipeline by reading best hyperparameters from `logreg_best_params.json`.
- `.github/workflows/hp_cml.yaml` uses CML GitHub Action to run a DVC pipeline and compare metrics between the training branch and main. This pipeline is triggered when you open a pull request against the main branch.










