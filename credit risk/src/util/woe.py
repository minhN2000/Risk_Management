# Weight of Evidence
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_woe( df,
                 id_column_name,
                 feature_column_name,
                 target_column_name,
                 sort_by_woe=True
                ):
    """
    This function computes the Weight of Evidence (WoE) and Information Value (IV) for a given feature.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    id_column_name (str): Name of the column to be used for aggregation.
    feature_column_name (str): Name of the feature for which WoE and IV should be computed.
    target_column_name (str): Name of the target variable.
    sort_by_woe (bool): If True, the output will be sorted by WoE.

    Returns:
    matrix (pd.DataFrame): A dataframe containing the WoE and IV for each category of the feature.
    """
    # contigency matrix
    matrix = pd.pivot_table(
        data=df,
        index=feature_column_name,
        columns=target_column_name,
        values=id_column_name,
        aggfunc=pd.Series.count,
    )
    label_columns = ["Non Default", "Default"]
    matrix.columns = label_columns

    # WOE
    matrix["Total Obs"] = matrix.sum(axis=1)
    for label in label_columns:
        matrix[f"% {label}"] = matrix[label] / matrix[label].sum()
    matrix["WoE"] = np.log(
        matrix[f"% {label_columns[1]}"] / matrix[f"% {label_columns[0]}"]
    )

    # IV
    matrix["IV"] = (
        matrix["WoE"]
        * (matrix[f"% {label_columns[1]}"] - matrix[f"% {label_columns[0]}"])
    ).sum()

    # sort by
    if sort_by_woe:
        matrix = matrix.sort_values(by="WoE")
    return matrix

def plot_woe_by_category(df, rotate=False):
    """
    This function plots the WoE for each category of a given feature.

    Parameters:
    df (pd.DataFrame): Dataframe containing the WoE for each category of the feature. It should be the output of the 'compute_woe' function.
    rotate (bool): If True, the x-axis labels will be rotated 90 degrees. 

    Returns:
    None
    """
    woe_column = "WoE"
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(df.index, df[woe_column], "o--", color="black")
    ax.set_xlabel(df.index.name.capitalize())
    if rotate:
        plt.xticks(rotation=90)
    ax.set_ylabel(woe_column)
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()

def display_woe(df,
                 id_column_name,
                 feature_column_name,
                 target_column_name,
                 sort_by_woe=True) :
    """
    This function computes the WoE and IV for a given feature, displays the result and plots the WoE for each category of the feature.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    id_column_name (str): Name of the column to be used for aggregation.
    feature_column_name (str): Name of the feature for which WoE and IV should be computed.
    target_column_name (str): Name of the target variable.
    sort_by_woe (bool): If True, the output will be sorted by WoE.

    Returns:
    None
    """
    woe_df = compute_woe(df,
                     id_column_name,
                     feature_column_name,
                     target_column_name)
    
    iv_val = woe_df['IV'].sum()
    print(f'IV value: {iv_val}')
    display(woe_df)
    plot_woe_by_category(woe_df.sort_index())