# Weight of Evidence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_woe( df: pd.DataFrame,
                 id_column_name: str,
                 feature_column_name,
                 target_column_name,
                 sort_by_woe=True
                ):

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

def plot_woe_by_category(df, rotate=False) -> None:
    woe_column = "WoE"
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(df.index, df[woe_column], "o--", color="black")
    ax.set_xlabel(df.index.name.capitalize())
    if rotate:
        plt.xticks(rotation=90)
    ax.set_ylabel(woe_column)
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()