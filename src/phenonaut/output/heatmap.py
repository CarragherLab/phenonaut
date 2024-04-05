# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.ticker import StrMethodFormatter


def write_heatmap_from_df(
    df: pd.DataFrame,
    title: str,
    output_file: Union[str, Path],
    axis_labels: tuple[str, str] = ("", ""),
    annot: Union[bool, np.ndarray] = True,
    transpose: Union[bool, None] = None,
    standard_deviations_df: Optional[pd.DataFrame] = None,
    figsize: Optional[Union[Tuple, List]] = None,
    figure_dpi: int = 300,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotation_fontsize: int = 10,
    annotation_best_fontsize: int = 12,
    lower_is_better: bool = False,
    annotation_format_string_value=".2f",
    annotation_format_string_std=".3f",
    highlight_best: bool = True,
    pallet_name: str = "seagreen",
    put_cbar_on_left: bool = False,
    sns_colour_palette: Optional[Colormap] = None,
) -> None:
    """Write heatmap from pd.DataFrame

    Write a PNG or SVG heatmap image to a file uisng seaborn to read from
    a pd.DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing values for the heatmap
    title : str
        Heatmap title
    output_file : Union[str, Path]
        Output path for the image, must have the extension .png or .svg.
    axis_labels : tuple[str, str], optional
        Tuple denoting labels to use for each axis, by default ("", "")
    annot : Union[bool, np.ndarray], optional
        If True, the the values of each heatmap cell/box are writen alongside
        applying the correct background colouring to the cell/box.
        By default True.
    transpose : Union[bool, None], optional
        Transpose the heatmap, by default None
    standard_deviations_df : Optional[pd.DataFrame], optional
        Optionally, supply a second pd.DataFrame containing standard deviations
        which may be included alongside the annotations within each cell/box.
        By default None.
    figsize : Optional[Union[Tuple, List]], optional
        Optional argument for setting the figure size. If None, then a target
        output width is set at 10, and the height calculated through examination
        of the number of elements, keeping square aspect ratios for each
        cell/box. By default None.
    figure_dpi : int, optional
        Resolution (dots per inch) at which to save the image (only useful if
        writing PNGs), by default 300.
    vmin : Optional[float], optional
        Minimum value to use for defining the colour/shading range. If None,
        then it is set to the lowest value present in the input DataFrame. By
        default None.
    vmax : Optional[float], optional
        Maximum value to use for defining the colour/shading range. If None,
        then it is set to the highest value present in the input DataFrame. By
        default None.
    annotation_fontsize : int, optional
        Font size for annotations, by default 10
    annotation_best_fontsize : int, optional
        Font size for the annotation used to highlight the best value present
        within the input DataFrame, by default 12
    lower_is_better : bool, optional
        Used for selecting the best value within the dataframe. If True, then
        the best heatmap value is taken to be the lowest. By default False.
    annotation_format_string_value : str, optional
        Optional formatting string for annotation f-strings, by default ".2f".
    annotation_format_string_std : str, optional
        Optional formatting string for standard deviation annotation f-strings,
        by default ".3f".
    highlight_best : bool, optional
        If True, then highlight the best result, with best being determined
        through use of the 'lower_is_better' keyword. By default True.
    pallet_name : str, optional
        Pallet to use for the colour bar. By default "seagreen".
    put_cbar_on_left : bool, optional
        If True, then position the colour bar on the left. If False, then
        position the colour bar on the right. By default False.
    sns_colour_palette : Optional[Colormap], optional
        Optionally, supply a seaborn colour map for use in the colour bar. By
        default None.
    """

    padding = {"top": 0.25, "bottom": 1.24, "element_height": 0.60}
    if figsize is None:
        figsize = (
            10.0,
            padding["top"]
            + padding["bottom"]
            + padding["element_height"] * df.shape[0],
        )
    if (transpose is None and df.shape[0] > df.shape[1]) or transpose:
        df = deepcopy(df.transpose())
    # Turn column and row labels with spaces into multiline items
    df = df.rename(
        columns={c: c.replace(" ", "\n") for c in df.columns if isinstance(c, str)},
        index={
            ind: ind.replace(" ", "\n").replace(":", "\n")
            for ind in df.index
            if isinstance(ind, str)
        },
    )

    if not isinstance(annot, np.ndarray):
        if annot and standard_deviations_df is not None:
            annot = (
                np.array(
                    [
                        f"{val:{annotation_format_string_value}}\n({std:{annotation_format_string_std}})"
                        for val, std in zip(
                            df.values.ravel(), standard_deviations_df.values.ravel()
                        )
                    ]
                )
                .reshape(df.values.shape)
                .tolist()
            )
        else:
            annot = (
                np.array(
                    [
                        f"{val:{annotation_format_string_value}}"
                        for val in df.values.ravel()
                    ]
                )
                .reshape(df.values.shape)
                .tolist()
            )

    fig, ax = plt.subplots(figsize=figsize, facecolor="w")
    if highlight_best:
        mask = df.values == np.nanmax(df.values)
        if lower_is_better:
            mask = ~mask
        negated_mask = ~mask
        sns.heatmap(
            df,
            cmap=sns.light_palette(pallet_name, as_cmap=True, reverse=lower_is_better)
            if sns_colour_palette is None
            else sns_colour_palette,
            mask=mask,
            annot=annot,
            linewidths=0.02,
            ax=ax,
            fmt="",
            annot_kws={"fontsize": annotation_fontsize},
            vmin=np.nanmin(df.values) if vmin is None else vmin,
            vmax=np.nanmax(df.values) if vmax is None else vmax,
            cbar_kws=dict(location="left") if put_cbar_on_left else None,
        )
        sns.heatmap(
            df,
            cmap=sns.light_palette(pallet_name, as_cmap=True, reverse=lower_is_better)
            if sns_colour_palette is None
            else sns_colour_palette,
            mask=negated_mask,
            annot=annot,
            linewidths=0.02,
            ax=ax,
            fmt="",
            annot_kws={
                "fontsize": annotation_best_fontsize,
                "style": "italic",
                "weight": "bold",
            },
            cbar=False,
            vmin=np.nanmin(df.values) if vmin is None else vmin,
            vmax=np.nanmax(df.values) if vmax is None else vmax,
            cbar_kws=dict(location="left") if put_cbar_on_left else None,
        )
    else:
        sns.heatmap(
            df,
            cmap=sns.light_palette(pallet_name, as_cmap=True, reverse=lower_is_better)
            if sns_colour_palette is None
            else sns_colour_palette,
            annot=annot,
            linewidths=0.02,
            ax=ax,
            fmt="",
            annot_kws={"fontsize": annotation_fontsize},
            vmin=np.nanmin(df.values) if vmin is None else vmin,
            vmax=np.nanmax(df.values) if vmax is None else vmax,
            cbar_kws=dict(location="left") if put_cbar_on_left else None,
        )

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    ax.set_title(title)
    plt.yticks(rotation=15)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(output_file, dpi=figure_dpi)
    plt.close()
