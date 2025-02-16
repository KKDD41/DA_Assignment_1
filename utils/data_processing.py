import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BRONZE_PATH = "./data/bronze/"
SILVER_PATH = "./data/silver/"
GOLD_PATH = "./data/gold/"

GROUP_FILENAME = "group_data.csv"
POSTS_FILENAME = "posts_data.csv"


def load_dataset(filepath: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(filepath, **kwargs)


def remove_duplicates_and_empty_rows(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    all_rows = len(df)

    df_cleaned = df.dropna(how='all')
    rows_wo_empty_rows = len(df_cleaned)

    df_cleaned = df.drop_duplicates()
    rows_wo_duplicates = len(df_cleaned)

    print(f"For dataset '{df_name}':")
    print(f"Number of duplicated rows: {all_rows - rows_wo_duplicates}.")
    print(f"Number of empty rows: {all_rows - rows_wo_empty_rows}. \n")
    return df_cleaned


def check_if_column_is_unique_per_row(df: pd.DataFrame, col_name: str) -> bool:
    number_of_rows_total = len(df)
    number_of_unique_values_in_col = df[
        col_name].nunique()

    return number_of_rows_total == number_of_unique_values_in_col


def predict_missing_values_with_linear_regression(
        mydf: pd.DataFrame,
        col_A: str,
        col_B: str,
        predicted_flag_col_name: str
):
    pd.options.mode.chained_assignment = None
    
    df = mydf.copy()

    scaler = StandardScaler()
    df[col_A] = scaler.fit_transform(df[[col_A]])

    train_df = df.dropna()
    model = LinearRegression()
    model.fit(train_df[[col_A]], train_df[col_B])

    missing_values_mask = df[col_B].isna()
    df.loc[missing_values_mask, col_B] = model.predict(df.loc[missing_values_mask, [col_A]])

    df[predicted_flag_col_name] = False
    df.loc[missing_values_mask, predicted_flag_col_name] = True

    df[col_B] = df[col_B].astype(int)
    df[col_A] = mydf[col_A]

    return df


def display_dataset_description(df: pd.DataFrame, df_name: str = ""):
    print(f" -- DATASET {df_name} INFO --")
    print(df.info())

    print(f" -- DATASET {df_name} DESCRIPTION --")
    print(df.description().T)
