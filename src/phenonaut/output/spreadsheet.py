# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import Union

import pandas as pd


def write_xlsx(
    output_file: Union[Path, str],
    dfs: Union[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame]],
    df_names: Union[tuple, str] = None,
):
    """Write an XLSX spreadsheet file

    Writing an XLSX file allows multiple sheets to be embedded in a single file
    which may be advantageous over CSV/TSV in many situations.

    Parameters
    ----------
    output_file : Union[Path, str]
        Output file path.
    dfs : Union[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame]]
        DataFrame for output, list of DataFrames, or dictionary of DataFrames.
        If a single DataFrame is given, or list of DataFrames, then the df_names
        argument must contain a str, or list of str respectively. If a
        dictionary is passed for this parameter, then the keys under which each
        DataFrame exist are used as sheet names.
    df_names : Union[tuple, str], optional
        The name of sheets under which DataFrames are stored in the spreadsheet.
        If a dictionary was given for the dfs parameter, then this parameter has
        no effect. If this parameter is None, then sheets are named 'Sheet#N',
        where #N is a number starting at 1 and incremented with each addition of
        a new sheet. By default None.

    Raises
    ------
    ValueError
        output_file should be Path or str
    """
    if isinstance(output_file, str):
        output_file = Path(output_file)
    if isinstance(output_file, Path):
        output_file = output_file.resolve()
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
    else:
        raise ValueError(
            f"output_file should have been a Path or str, it was {type(output_file)}"
        )

    if isinstance(dfs, dict):
        df_names = list(dfs.keys())
        dfs = list(dfs.values())

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    if df_names is None:
        df_names = [f"Sheet{i+1}" for i in range(len(dfs))]
    if isinstance(df_names, str):
        df_names = [df_names]
    if len(dfs) > len(df_names):
        num_to_add = len(dfs) - len(df_names)
        df_names.extend(
            [f"Sheet{i+1}" for i in range(len(df_names), len(df_names) + num_to_add)]
        )

    with pd.ExcelWriter(output_file) as writer:
        for df, sheet_name in zip(dfs, df_names):
            df.to_excel(writer, sheet_name=sheet_name)
