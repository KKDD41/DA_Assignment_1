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
) -> pd.DataFrame:
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


def get_outliers_IRQ(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    q1 = df[col_name].quantile(0.25)
    q2 = df[col_name].quantile(0.75)

    IRQ = q2 - q1

    outliers_df = df[
        (df[col_name] < (q1 - 1.5 * IRQ)) | (df[col_name] > (q2 + 1.5 * IRQ))
    ]

    num_outliers = len(outliers_df)
    print(f"Number of outliers in '{col_name}': {num_outliers}")
    if num_outliers > 0:
        print("Outliers statistics:")
        print(outliers_df[col_name].describe())
    else:
        print(f"No outliers detected in {col_name}.")

    return outliers_df


def get_distinct_rows_from_dataframes(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    combined_df = pd.concat(dfs, keys=range(len(dfs)))

    duplicated = combined_df.duplicated(keep=False, subset=combined_df.columns)
    distinct_combined_df = combined_df[~duplicated]

    distinct_dfs = [group.droplevel(0) for _, group in distinct_combined_df.groupby(level=0)]

    return distinct_dfs