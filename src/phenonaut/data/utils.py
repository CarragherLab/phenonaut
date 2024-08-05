import phenonaut
from uuid import uuid4


def dataset_intersection(
    datasets: list[phenonaut.data.Dataset], groupby: str | list[str], inplace=False
):
    """Perform intersection of datasets on common column values

    This is useful to match experimental data using a groupby of ['cpd', 'conc'], datasets can be
    filtered to contain only compounds at concentrations present in all dataframes.  This is useful
    in integration work where each dataset represents a different view/assay technology]

    Parameters
    ----------
    datasets : list[phenonaut.data.Dataset]
        List of datasets to perform treatment intersection filtering
    groupby : str | list[str]
        Columns present in each dataset which which rows must match across all datasets

    Returns
    -------
    list[phenonaut.data.Dataset] | None
        List of filtered datasets if inplace is False, else datasets are altered in place and None
        is returned

    Raises
    ------
    ValueError
        Error if groupby fields are not found in all Dataset DataFrame columns
    RuntimeError
        Temporary column exists in dataframe.  This error should not ever occur
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    if not inplace:
        datasets = [ds.copy() for ds in datasets]

    intersection_column_name = f"tmp_{uuid4()}"
    for ds in datasets:
        for gb in groupby:
            if gb not in ds.df.columns:
                raise ValueError(
                    f"The groupby field '{gb}' was not found in the dataset {ds}"
                )
        if intersection_column_name in ds.df.columns:
            raise RuntimeError(
                f"Should not have happened, a unique temp column name '{intersection_column_name}' was found in the dataset {ds}"
            )

    if len(datasets) == 1:
        return datasets

    gb_df = (
        datasets[0]
        .df[groupby]
        .drop_duplicates()
        .merge(datasets[1].df[groupby].drop_duplicates(), on=groupby)
    )

    for ds_i in range(2, len(datasets)):
        gb_df = gb_df.merge(datasets[ds_i].df[groupby].drop_duplicates(), on=groupby)

    gb_df[intersection_column_name] = "inall"

    for ds_i in range(len(datasets)):
        datasets[ds_i].df = (
            datasets[ds_i]
            .df.merge(gb_df, on=groupby)
            .dropna(subset=[intersection_column_name])
            .drop(columns=intersection_column_name)
        )

    if not inplace:
        return datasets
