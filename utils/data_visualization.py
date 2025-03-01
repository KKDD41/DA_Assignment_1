import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_barchart(df: pd.DataFrame, categorical_col_name: str | list[str]):
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("viridis", n_colors=df[categorical_col_name].nunique())

    sns.countplot(
        y=categorical_col_name,
        data=df,
        palette=palette,
        hue=categorical_col_name,
        legend=False
    )

    plt.title(f'Distribution of {categorical_col_name}', fontsize=16)
    plt.xlabel('Counts', fontsize=14)
    plt.ylabel('Categories', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_histogram_and_boxplot(df: pd.DataFrame, numerical_col_name: str):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))

    sns.histplot(df[numerical_col_name], kde=True, color='skyblue', ax=axes[0, 0])
    axes[0, 0].set_title(f'Distribution of {numerical_col_name}', fontsize=16)
    axes[0, 0].set_xlabel(numerical_col_name, fontsize=14)
    axes[0, 0].set_ylabel('Frequency', fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    sns.boxplot(x=df[numerical_col_name], color='lightgreen', ax=axes[0, 1])
    axes[0, 1].set_title(f'Boxplot of {numerical_col_name}', fontsize=16)
    axes[0, 1].set_xlabel(numerical_col_name, fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    sns.histplot(df[numerical_col_name],
                 kde=True,
                 color='deepskyblue',
                 ax=axes[1, 0],
                 edgecolor='black',
                 linewidth=1.5,
                 stat="percent")
    axes[1, 0].set_title(f'Distribution of {numerical_col_name}', fontsize=16)
    axes[1, 0].set_xlabel(numerical_col_name, fontsize=14)
    axes[1, 0].set_ylabel('Percent', fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)


    sns.boxplot(x=df[numerical_col_name],
                color='mediumseagreen',
                ax=axes[1, 1],
                linewidth=2.5)
    axes[1, 1].set(xscale="log")
    axes[1, 1].set_title(f'Boxplot of {numerical_col_name} (Log Scale)', fontsize=14)
    axes[1, 1].set_xlabel(numerical_col_name, fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrices(df: pd.DataFrame, cols_to_remove: list[str]):
    sns.set(style="white", context='talk')

    numerical_cols = list(df.select_dtypes(include=['number']).columns)
    for col in cols_to_remove:
        numerical_cols.remove(col)

    numerical_data = df[numerical_cols]

    pearson_corr = numerical_data.corr(method='pearson')
    spearman_corr = numerical_data.corr(method='spearman')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 14))

    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .8}, ax=axes[0],
                linewidths=.5, linecolor='black')
    axes[0].set_title('Pearson Correlation Matrix', fontsize=20)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .8}, ax=axes[1],
                linewidths=.5, linecolor='black')
    axes[1].set_title('Spearman Correlation Matrix', fontsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_scatter_graph(df: pd.DataFrame, x_column: str, y_columns: list[str]):
    sns.set(style="whitegrid", context='talk')
    plt.figure(figsize=(15, 10))

    for i, y_column in enumerate(y_columns):
        sns.scatterplot(data=df, x=x_column, y=y_column, label=y_column, s=(i + 1) * 50)

    plt.legend(title="Features:")
    plt.title(f'Scatter Plot of {x_column} vs. {", ".join(y_columns)}', fontsize=20)
    plt.xlabel(x_column, fontsize=16)
    plt.ylabel('Values', fontsize=16)

    plt.show()


def analyse_categorical_features(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        category_counts = df[col].value_counts(dropna=False)
        print(f"Counts for {col}:\n{category_counts}\n")

        total_counts = len(df[col])
        rare_categories = category_counts[category_counts < (0.01 * total_counts)]
        if not rare_categories.empty:
            print(f"Rare categories in {col}:\n{rare_categories}\n")

        plot_barchart(df, col)

        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"Missing values in {col}: {missing_count}.\n")
        else:
            print(f"No missing values in {col}.\n")


def analyse_numerical_features(
        df: pd.DataFrame,
        cols_to_remove: list[str],
        show_statistics: bool = True
):
    numerical_cols = list(df.select_dtypes(include=['number']).columns)

    for col in cols_to_remove:
        numerical_cols.remove(col)

    if not numerical_cols:
        print("No numerical features exists in dataset.")

    for col in numerical_cols:
        if show_statistics:
            print(f"Descriptive Statistics for {col}:")
            print(df[col].describe())
            print("\n")

        plot_histogram_and_boxplot(df, col)

        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"Missing values in {col}: {missing_count}.\n")
        else:
            print(f"No missing values in {col}.\n")


def analyse_numerical_features_per_category(df: pd.DataFrame, cols_to_remove: list[str]):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = list(df.select_dtypes(include=['number']).columns)

    for col in cols_to_remove:
        numerical_cols.remove(col)

    for col in categorical_cols:
        print(f"Graphs displayed for category {col}: \n")
        categories = list(df[col].unique())

        fig, axes = plt.subplots(nrows=len(categories), ncols=len(numerical_cols), figsize=(20, 20))

        for i, category in enumerate(categories):
            for j, numerical_col_name in enumerate(numerical_cols):
                sns.histplot(df[numerical_col_name], kde=True, color='skyblue', ax=axes[i, j])
                axes[i, j].set_title(f'[{category}] Distribution of {numerical_col_name}', fontsize=10)
                axes[i, j].set_xlabel(numerical_col_name, fontsize=8)
                axes[i, j].set_ylabel('Frequency', fontsize=8)
                axes[i, j].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


def analyse_aggregated_features_per_category(
        df: pd.DataFrame,
        agg_cols: list[list[str]],
        metrics: list[str],
        agg_method: str = 'sum'
):
    grouped_dfs = []
    for agg_col_list in agg_cols:
        if agg_method == 'sum':
            grouped_df = df[
                metrics + agg_col_list
            ].groupby(agg_col_list).sum()
        elif agg_method == 'mean':
            grouped_df = df[
                metrics + agg_col_list
            ].groupby(agg_col_list).mean()

        grouped_dfs.append(grouped_df)

    cols_num = len(agg_cols)
    rows_num = len(metrics)

    fig, axes = plt.subplots(nrows=rows_num, ncols=cols_num, figsize=(20, 45))
    axes = axes.flatten()

    for i in range(0, rows_num * cols_num, cols_num):
        metric = metrics[i // cols_num] # reach

        for j in range(cols_num):
            x = agg_cols[j][0]
            hue = agg_cols[j][min(1, len(agg_cols[j]) - 1)]

            sns.barplot(
                data=grouped_dfs[j].reset_index(),
                hue=hue,
                y=metric,
                x=x,
                ax=axes[i + j]
            )
            axes[i + j].set_title(f'{agg_method[0].upper() + agg_method[1:]} of [{metric.upper()}] by [{", ".join([x.upper() for x in agg_cols[j]])}]')
            axes[i + j].set_xlabel(x)
            axes[i + j].set_ylabel(f'{agg_method[0].upper() + agg_method[1:]} of ' + metric)

    plt.savefig(f"{random.randint(1000, 2000)}.png")

    plt.tight_layout()
    plt.show()




def plot_linechart(
        df: pd.DataFrame,
        x_col_name: str,
        y_col_name: str,
        grouping_col: str,
        ax
):
    if grouping_col:
        lineplot = sns.lineplot(
            data=df,
            x=x_col_name,
            y=y_col_name,
            hue=grouping_col,
            style=grouping_col,
            markers=True,
            dashes=False,
            ax=ax
        )

        plt.title(f'[{y_col_name.upper()}] over time by [{grouping_col.upper()}]', fontsize=16)
        plt.xlabel(x_col_name, fontsize=14)
        plt.ylabel(y_col_name, fontsize=14)
        plt.xticks(rotation=45)
        lineplot.legend(title=f'[{grouping_col.upper()}]')
    else:
        lineplot = sns.lineplot(
            data=df,
            x=x_col_name,
            y=y_col_name,
            markers=True,
            dashes=False,
            ax=ax
        )

        plt.title(f'[{y_col_name.upper()}] over time', fontsize=16)
        plt.xlabel(x_col_name, fontsize=14)
        plt.ylabel(y_col_name, fontsize=14)
        plt.xticks(rotation=45)



def analyse_features_trend(df: pd.DataFrame, x_col_name: str, grouping_col: str, in_one_row: bool = False):
    numerical_cols = list(df.select_dtypes(include=['number']).columns)

    if not in_one_row:
        rows_num = len(numerical_cols)
        cols_num = 2
    else:
        rows_num = len(numerical_cols) * 2
        cols_num = 1

    fig, axes = plt.subplots(nrows=rows_num, ncols=cols_num, figsize=(20, 45))
    axes = axes.flatten()

    for i in range(0, rows_num * cols_num, max(cols_num, 2)):
        if not in_one_row:
            plot_linechart(df, x_col_name, numerical_cols[i // cols_num], '', axes[i])
            plot_linechart(df, x_col_name, numerical_cols[i // cols_num], grouping_col, axes[i + 1])
        else:
            plot_linechart(df, x_col_name, numerical_cols[i // 2 * cols_num], '', axes[i])
            plot_linechart(df, x_col_name, numerical_cols[i // 2 * cols_num], grouping_col, axes[i + 1])

    plt.tight_layout()

    plt.savefig(f"{random.randint(1, 999)}.png")
    plt.show()