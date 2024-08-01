import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils_and_constants import PROCESSED_DATASET, RAW_DATASET, TARGET_COLUMN



def read_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, header=None)
    return df



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Replace the '?'s with NaN in dataset
    df_nans_replaced = df.replace("?", np.nan)

    # Create a copy of the NaN replacement DataFrame
    df_imputed = df_nans_replaced.copy()

    # Iterate over each column of cc_apps_nans_replaced and impute the most frequent value for object data types and the mean for numeric data types
    for col in df_imputed.columns:
        # Check if the column is of object type
        if df_imputed[col].dtypes == "object":
            # Impute with the most frequent value
            df_imputed[col] = df_imputed[col].fillna(
                df_imputed[col].value_counts().index[0]
            )
        else:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    # Dummify the categorical features
    df_encoded = pd.get_dummies(df_imputed, drop_first=True, dtype=int)

    return df_encoded



def scale_data(df_encoded_features: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df_encoded_features.columns = df_encoded_features.columns.astype(str)
    X_preprocessed = scaler.fit_transform(df_encoded_features)
    return pd.DataFrame (X_preprocessed, columns=df_encoded_features.columns)



def main():
    # Read data
    cc_apps = read_dataset(filename= RAW_DATASET)

    # clean dataset
    cc_apps_cleaned = clean_dataset(cc_apps)
    print(cc_apps_cleaned)
    # scale data
    cc_apps_features = cc_apps_cleaned.iloc[:, :-1]
    cc_apps_scaled = scale_data(cc_apps_features)
     
    # write processed dataset
    cc_apps_labels = cc_apps_cleaned.iloc[:, [-1]]
    cc_apps = pd.concat([cc_apps_scaled,cc_apps_labels], axis=1)
    cc_apps.to_csv(PROCESSED_DATASET, index=None)

if __name__ == "__main__":
    main()




  

