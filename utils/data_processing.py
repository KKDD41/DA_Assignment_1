import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BRONZE_PATH = "./data/bronze/"
SILVER_PATH = "./data/silver/"
GOLD_PATH = "./data/gold/"

GROUP_FILENAME = "group_data"
POSTS_FILENAME = "posts_data"


def load_dataset(filepath: str, file_format: str = 'csv', **kwargs) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(filepath + ".csv", **kwargs)
    elif file_format == "parquet":
        return pd.read_parquet(filepath + ".parquet", **kwargs)
    else:
        return None


def save_dataset(df: pd.DataFrame, filepath: str, **kwargs):
    df.to_parquet(filepath, **kwargs)


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


def predict_next_month_average_ARIMA(df: pd.DataFrame,
                                     column_name: str,
                                     use_auto_arima: bool,
                                     information_criterion: str = 'aic',
                                     order=(0, 0, 0)):
    df = df.sort_values('post_date')

    time_series = df.groupby('post_date')[column_name].mean()

    print('DF test results:')
    df_result = adfuller(time_series)
    df_labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for result_value, label in zip(df_result, df_labels):
        print(label + ' : ' + str(result_value))

    if df_result[1] <= 0.05:
        print("Time series is stationary.\n")
    else:
        print("Time series is not stationary.\n")

    time_series2 = time_series.copy()

    if use_auto_arima:
        arima_params = auto_arima(
            y=time_series2,
            trace=True,
            stationary=True,
            max_order=None,
            maxiter=1000,
            information_criterion=information_criterion
        )

        arima_params = str(arima_params)
        print(arima_params[7: 12])
        order = tuple(map(int, arima_params[7: 12].split(',')))

    model = ARIMA(time_series, order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)

    plt.figure(figsize=(12, 6))

    plt.plot(time_series, label='Historical Daily Average')
    plt.plot(forecast, label='Forecasted Daily Average', color='red')

    plt.title(f'Forecasting Daily Average {column_name.capitalize()} for the Next Month')
    plt.xlabel('Date')
    plt.ylabel(column_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

    return pd.DataFrame({f'Forecasted Average {column_name}': forecast})


def predict_next_month_average_QR(df, response_var, quantile, predictors):
    df = df.sort_values('post_date')
    df = df.groupby('post_date').mean()

    formula = f"{response_var} ~ {' + '.join(predictors)}"

    mod = smf.quantreg(formula, df)
    res = mod.fit(q=quantile)

    print(res.summary())

    coefficients = res.params

    future_values = coefficients['Intercept']
    for predictor in predictors:
        future_values += coefficients[predictor] * df[predictor].mean()

    print(future_values)
    plt.figure(figsize=(12, 6))

    plt.plot(df, label='Historical Daily Average')
    plt.plot(future_values, label='Forecasted Daily Average', color='red')

    plt.title(f'Forecasting Daily Average {response_var.capitalize()} for the Next Month')
    plt.xlabel('Date')
    plt.ylabel(response_var.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()
    return future_values


def predict_next_month_average_SW(df, column_name, window_size, forecast_days=30):
    df = df.groupby('post_date')[[column_name]].mean()
    df['moving_average'] = df[column_name].rolling(window=window_size, min_periods=1).mean()

    last_average = df['moving_average'].iloc[-1]
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_values = [last_average] * forecast_days

    forecast_df = pd.DataFrame({column_name: forecast_values}, index=forecast_dates)
    forecast_df['moving_average'] = last_average

    combined_df = pd.concat([df, forecast_df])

    plt.figure(figsize=(12, 6))
    plt.plot(combined_df[column_name], label='Actual Values', alpha=0.6)
    plt.plot(combined_df['moving_average'], label='Moving Average', color='red', linestyle='--')
    plt.plot(forecast_df.index, forecast_df[column_name], label='Forecast', color='green')
    plt.title(f'Forecasting {column_name.capitalize()} for the Next {forecast_days} Days')
    plt.xlabel('Date')
    plt.ylabel(column_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"The final predicted moving average value for {column_name} is {last_average}")

    return combined_df
