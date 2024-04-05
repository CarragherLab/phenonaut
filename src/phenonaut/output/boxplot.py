# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def write_boxplot_to_file(
    df: pd.DataFrame,
    x_label_in_df: str,
    y_label_in_df: str,
    output_file: Union[str, Path],
    title="Boxplot",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize=(10, 6.18),
    orient: Optional[str] = None,
):
    """Write a boxplot to a file

    Boxplot generation from pd.DataFrames using the Seaborn library

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    x_label_in_df : str
        Label for the x-axis which appears in the dataframe.
    y_label_in_df : str
        Label for the y-axis which appears in the dataframe.
    output_file : Union[str, Path]
        Output (PNG) filename.
    title : str, optional
        Title to be displayed on the plot, by default "Boxplot"
    x_label : Optional[str], optional
        Optionally, override the use of the x-label present in the dataframe
        and display a different one on the axis. If None, the the value of
        x_label_in_df is used. By default None.
    y_label : Optional[str], optional
        Optionally, override the use of the y-label present in the dataframe
        and display a different one on the axis. If None, the the value of
        y_label_in_df is used. By default None.
    figsize : tuple, optional
        Optional tuple of image height-width (in inches - as dictated by
        matplotlib/seaborn), by default (10, 6.18).
    orient : str, optional
        'h' or 'v' accepted and passed to seaborn to orient boxplots
        horizontally or vertically respectively.  If None, then the
        best orientation is guessed.  By default None.
    """
    sns.set(rc={"figure.figsize": figsize})
    tidy_df = df.replace(" ", r"\n", regex=True)

    if isinstance(output_file, str):
        output_file = Path(output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    ax = sns.boxplot(x=x_label_in_df, y=y_label_in_df, data=tidy_df, orient=orient)
    if x_label is not None:
        ax.set(xlabel=x_label)
    if y_label is not None:
        ax.set(ylabel=y_label)

    ax.set_title(title)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(output_file)
    plt.close()
