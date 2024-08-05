import phenonaut
import numpy as np
import pandas as pd
from tqdm import tqdm


def concatenate_datasets_horizontally(
    datasets: list[phenonaut.data.Dataset],
    merge_field: str | list[str] | None = None,
    how: str = 'PerfectMatch',
    n_random_right: int = 4,
    random_state: int | np.random.Generator = 7,
    quiet: bool = True,
):
    """Concatenate datasets horizontally

    Useful for merging two or more datasets and expanding their features by a factor of the number
    of datesets.  This function expands columns, not rows.

    At present, only one concatenation method is implemented:

    - 'EnumerateAll'
        - - For each treatment, all data combinations are enumerated for the left and right datasets
        (iterating through a list using left and right if concatenating more than 2). This has the
        effect of massively increasing the number of samples in the final dataset, as merging
        treatments from 2 datasets where replicate cardinalities are 4 in both, results in the
        new dataset having a replicate cardinality of 16 (4x4). Note that all non-essential columns
        in the dataframe will be removed when using this approach. This is a design decision taken
        with the aim of controlling spiraling memory requirements when performing multiple
        concatenations.
    - 'EnumerateThenMatchCardinality'
        - For each treatment, all data combinations are enumerated for the left and right datasets
        (iterating through a list using left and right if concatenating more than 2).  Then, a
        sample is taken from all of these combinations to bring the cardinality down to that of the
        # left dataset group treatments. Warning, this approach with downsampling causes smoothing of
        the dataset, removing outliers and may result in artificially high benchmark scores.

    - 'PerfectMatch'
        - For each treatment group on the left, match with the treatment group on the right. If
        group cardinalities are not the same, then drop samples until they match.


    Parameters
    ----------
    datasets : list[phenonaut.data.Dataset]
        List of phenonaut Datasets which should be merged
    merge_field : str | list[str] | None
        The column name which is used to match treatments.  If None, then this is taken to be the
        perturbation_column from the first dataset, by default None
    how : str, optional
        Concatenation method, see function description above, by default 'PerfectMatch'
    n_random_right : int, optional
        Num sampled treatments to merge, see function description above, by default 4
    random_state : int | np.random.Generator, optional
        Random state as either an int for sampling, can also be a np.random.Generator, by default 7

    Returns
    -------
    phenonaut.data.Dataset
        Dataset made of horizontally concatenated datasets

    Raises
    ------
    NotImplementedError
        Requested concatenation method is not yet implemented
    """
    if merge_field is None:
        merge_field = datasets[0].perturbation_column
    if not isinstance(merge_field, list):
        merge_field = [merge_field]

    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    if how == 'EnumerateThenMatchCardinality':
        df = datasets[0].df.rename(
            columns={
                c: f"conchoriz_ds1_{c}"
                for c in datasets[0].df.columns
                if c not in merge_field
            }
        )
        new_ds_features = [f"conchoriz_ds1_{f}" for f in datasets[0].features]
        for i in range(1, len(datasets)):
            df = (
                df.merge(
                    datasets[i].df.rename(
                        columns={
                            c: f"conchoriz_ds{i+1}_{c}"
                            for c in datasets[i].df.columns
                            if c not in merge_field
                        }
                    ),
                    left_on=merge_field,
                    right_on=merge_field,
                    how='left',
                    suffixes=("", "_y"),
                )
                .groupby(merge_field)
                .apply(
                    lambda x: x.sample(int(np.sqrt(len(x))), random_state=random_state)
                )
                .reset_index(drop=True)
            )
            new_ds_features.extend(
                [f"conchoriz_ds{i+1}_{f}" for f in datasets[i].features]
            )
        df = df.query(
            " and ".join([f"not {field}.isnull() " for field in merge_field]),
            engine='python',
        )
    elif how == 'EnumerateAll':
        df = (
            datasets[0]
            .df[merge_field + datasets[0].features]
            .rename(
                columns={
                    c: f"conchoriz_ds1_{c}"
                    for c in datasets[0].df.columns
                    if c not in merge_field
                }
            )
        )
        new_ds_features = [f"conchoriz_ds1_{f}" for f in datasets[0].features]
        for i in range(1, len(datasets)):
            df = df.merge(
                datasets[i]
                .df[merge_field + datasets[i].features]
                .rename(
                    columns={
                        c: f"conchoriz_ds{i+1}_{c}"
                        for c in datasets[i].df.columns
                        if c not in merge_field
                    }
                ),
                left_on=merge_field,
                right_on=merge_field,
                how='left',
                suffixes=("", "_y"),
            ).reset_index(drop=True)
            new_ds_features.extend(
                [f"conchoriz_ds{i+1}_{f}" for f in datasets[i].features]
            )
        # df=df.query(" and ".join([f"not {field}.isnull() " for field in merge_field]), engine='python')
    elif how == 'PerfectMatch':
        if len(merge_field) > 1:
            raise ValueError(
                "Cannot perform concatenation with the PerfectMatch approach when multiple fields are required to match replicate groups. A simple workaround would be to make a new column concatenating treatmentID + dose fields, or whatever the required fields are"
            )
        merge_field = merge_field[0]
        treatments = datasets[0].get_unique_treatments()
        df = datasets[0].df.rename(
            columns={
                c: f"conchoriz_ds1_{c}"
                for c in datasets[0].df.columns
                if c != merge_field
            }
        )
        new_ds_features = [f"conchoriz_ds1_{f}" for f in datasets[0].features]
        for i in tqdm(
            range(1, len(datasets)), disable=quiet, desc="Concatenating datasets"
        ):
            added_features_for_this_dataset = False
            ds_right_df_renamed = datasets[i].df.rename(
                columns={c: f"conchoriz_ds{i+1}_{c}" for c in datasets[i].df.columns}
            )
            merged_df = pd.DataFrame()

            for trt in tqdm(
                treatments, disable=quiet, desc="Iterating groups", leave=False
            ):
                left = df.query(
                    f"{merge_field}=='{trt}'"
                    if isinstance(trt, str)
                    else f"{merge_field}=={trt}"
                )
                right = ds_right_df_renamed.query(
                    f"conchoriz_ds{i+1}_{merge_field}=='{trt}'"
                    if isinstance(trt, str)
                    else f"conchoriz_ds{i+1}_{merge_field}=={trt}"
                )
                if len(right) == 0:
                    continue
                if not added_features_for_this_dataset:
                    new_ds_features.extend(
                        [f"conchoriz_ds{i+1}_{f}" for f in datasets[i].features]
                    )
                    added_features_for_this_dataset = True
                if len(right) < len(left):
                    left = left.sample(len(right), random_state=random_state)
                if len(right) > len(left):
                    right = right.sample(len(left), random_state=random_state)
                leftright = pd.DataFrame(
                    np.hstack([left, right]),
                    columns=left.columns.tolist() + right.columns.tolist(),
                    index=left.index,
                )
                # leftright=pd.concat([left, right], axis=1, ignore_index=True)
                # leftright.columns=left.columns.tolist()+right.columns.tolist()
                merged_df = pd.concat([merged_df, leftright], axis=0)
            df = merged_df
        # df=df.query(f"not {merge_field}.isnull()", engine='python')

    else:
        raise NotImplementedError(
            f"Concatenation method '{how}' not yet implemented.  Implemented methods are: 'PairRightRandom'"
        )
    new_ds = phenonaut.data.Dataset(
        f"Concatednated ({', '.join([ds.name for ds in datasets])})",
        df,
        features=new_ds_features,
    )
    if isinstance(merge_field, list):
        new_ds.perturbation_column = (
            merge_field[0] if len(merge_field) == 1 else merge_field
        )
    else:
        new_ds.perturbation_column = merge_field

    new_ds.df = new_ds.df.reset_index()

    return new_ds


# elif how == 'PerfectMatch':
#         if len(merge_field)>1:
#             raise ValueError("Cannot perform concatenation with the PerfectMatch approach when multiple fields are required to match replicate groups. A simple workaround would be to make a new column concatenating treatmentID + dose fields, or whatever the required fields are")
#         merge_field=merge_field[0]
#         df=datasets[0].df.rename(columns={c:f"conchoriz_ds1_{c}" for c in datasets[0].df.columns if c not in merge_field})
#         new_ds_features=[f"conchoriz_ds1_{f}" for f in datasets[0].features]
#         for i in tqdm(range(1,len(datasets)), disable=quiet, desc="Concatenating datasets"):

#             new_ds_features.extend([f"conchoriz_ds{i+1}_{f}" for f in datasets[i].features])
#             merged_df=pd.DataFrame()
#             for name, grp in tqdm(df.groupby(merge_field), disable=quiet, desc="Iterating groups", leave=False, total=len(df[merge_field].unique())):
#                 _tmp_res=datasets[i].df.rename(columns={c:f"conchoriz_ds{i+1}_{c}" for c in datasets[i].df.columns}).query(f"conchoriz_ds{i+1}_{merge_field}=={name}" if isinstance(name, int) else f"conchoriz_ds{i+1}_{merge_field}=='{name}'")
#                 if len(_tmp_res)==0:
#                     continue
#                 if len(_tmp_res)<len(grp):
#                     grp=grp.sample(len(_tmp_res), random_state=random_state)
#                 if len(_tmp_res)>len(grp):
#                     _tmp_res=_tmp_res.sample(len(grp), random_state=random_state)
#                 merged_df=pd.concat([merged_df,pd.concat([grp, _tmp_res], axis=1)], axis=0, ignore_index=True)
#             df=merged_df
#         df=df.query(f"not {merge_field}.isnull()", engine='python')

#     else:
#         raise NotImplementedError(f"Concatenation method '{how}' not yet implemented.  Implemented methods are: 'PairRightRandom'")
#     new_ds=phenonaut.data.Dataset(f"Concatednated ({', '.join([ds.name for ds in datasets])})", df, features=new_ds_features)
#     if isinstance(merge_field, list):
#         new_ds.perturbation_column=merge_field[0] if len(merge_field)==1 else merge_field
#     else:
#         new_ds.perturbation_column=merge_field

#     new_ds.df=new_ds.df.reset_index()

#     return new_ds
