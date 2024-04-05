# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.errors import DataError

from phenonaut import Phenonaut, data, output
from phenonaut.data import Dataset

from .visualisation_base import PhenonautVisualisation


class Scatter(PhenonautVisualisation):
    """Phenonaut scatter visualisation object.

    The object should be constructed, and then its add member function used
    to add points to the scatter.  It may then be saved by calling save, or
    displayed using show.

    Scatter inherits from the PhenonautVisualisation base class, which
    optionally stores a dictionary called plot_config. As currently
    implemented, plot_config dictionaries can contain one nested dictionary
    under a 'plot_markers' key, with keys to this nested dictionary
    being  perturbation ids, and values are arguments to the matplotlib
    marker types. To populate this base class dictionary, pass a config
    dictionary to the constructor, like

    .. code-block:: python

        {'plot_markers':{'pert1':'X', 'pert2':'x'}}.

    Parameters
    ----------
    plot_config : Optional[Union[Path, str, dict]], optional
        Optional configuration dictionary, allows  alows specification of initialisation arguments via dictionary.
        Supply keys and values as argument as values. By default None.
    figsize : Optional[Tuple[float, float]], optional
        Output figure size (in inches, as directed by matplotlib/seaborn).
        By default (8, 6)
    title : Optional[str], optional
        Plot title, by default "2D scatter"
    x_label : Optional[str], optional
        x-axis label, by default None
    y_label : Optional[str], optional
        y-axis label, by default None
    show_legend : Optional[bool], optional
        If True, then add a legend to the plot. By default True.
    marker_size : int, optional
        The size of each datapoint within the scatter. By default 90
    axis_ranges : Optional[Tuple[Tuple[float, float], Tuple[float, float]]], optional
        Optional tuple of tuples giving the min and max extents of each axis.
        If None is given for any value, then the corresponding min/max for
        that axis is calcaulated and used.
        By default ((None, None), (None, None)).
    """

    def __init__(
        self,
        plot_config: Optional[Union[Path, str, dict]] = None,
        figsize: Optional[Tuple[float, float]] = (8, 6),
        title: Optional[str] = "2D scatter",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        show_legend: Optional[bool] = True,
        marker_size: int = 90,
        axis_ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
            (None, None),
            (None, None),
        ),
    ):
        """Phenonaut scatter visualisation object.

        The object should be constructed, and then its add member function used
        to add points to the scatter.  It may then be saved by calling save, or
        displayed using show.

        Scatter inherits from the PhenonautVisualisation base class, which
        optionally stores a dictionary called plot_config. As currently
        implemented, plot_config dictionaries can contain one nested dictionary
        under a 'plot_markers' key, with keys to this nested dictionary
        being  perturbation ids, and values are arguments to the matplotlib
        marker types. To populate this base class dictionary, pass a config
        dictionary to the constructor, like

        .. code-block:: python

            {'plot_markers':{'pert1':'X', 'pert2':'x'}}.

        Parameters
        ----------
        plot_config : Optional[Union[Path, str, dict]], optional
            Optional configuration dictionary, allows  alows specification of initialisation arguments via dictionary.
            Supply keys and values as argument as values. By default None.
        figsize : Optional[Tuple[float, float]], optional
            Output figure size (in inches, as directed by matplotlib/seaborn).
            By default (8, 6)
        title : Optional[str], optional
            Plot title, by default "2D scatter"
        x_label : Optional[str], optional
            x-axis label, by default None
        y_label : Optional[str], optional
            y-axis label, by default None
        show_legend : Optional[bool], optional
            If True, then add a legend to the plot. By default True.
        marker_size : int, optional
            The size of each datapoint within the scatter. By default 90
        axis_ranges : Optional[Tuple[Tuple[float, float], Tuple[float, float]]], optional
            Optional tuple of tuples giving the min and max extents of each axis.
            If None is given for any value, then the corresponding min/max for
            that axis is calcaulated and used.
            By default ((None, None), (None, None)).
        """
        super().__init__(plot_config)
        # Add arguments to figure config
        for var in [
            x
            for x in self.__init__.__code__.co_varnames
            if x not in ("self", "figure_config", "var") and x not in self.config.keys()
        ]:
            self.config[var] = eval(var)

        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.marker_size = marker_size

    def add(
        self,
        dataset: Union[Dataset, Phenonaut],
        perturbations: Optional[List[str]] = None,
        markers: Union[dict, list, bool] = True,
        marker_size: int = None,
    ):
        """Add points to the scatter plot

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            Phenonaut dataset or Phenonaut object (if it contains only one
            dataset) from which to take datapoints.
        perturbations : Optional[List[str]], optional
            If perturbation column is set on the dataset and a list of
            perturbations is given here, then plot only those matching
            datapoints.  If None, then all datapoints are plotted.
            By default None,
        markers : Union[dict, list, bool], optional
            Dictionary containing pertubation ids as keys, and then values to
            pass to matplotlib defining marker styles, or a list of tuples of
            this form. If True, then let matplotlib enumerate marker types.
            By default True.
        marker_size : int, optional
            Size at which to draw datapoints. If None, then the value is taken
            from the marker_size argument used to construct the scatter object
            see constructor docstring. By default None.

        Raises
        ------
        DataError
            Scatter plots are 2D, more than 2 features found.
        """
        if isinstance(dataset, Phenonaut):
            dataset = dataset[-1]
        if len(dataset.features) != 2:
            raise DataError(f"Scatter requires 2 features, got {len(dataset.features)}")
        if self.config.get("xlabel") is None:
            self.config["xlabel"] = dataset.features[0]
        if self.config.get("ylabel") is None:
            self.config["ylabel"] = dataset.features[1]

        if marker_size is None:
            marker_size = self.marker_size

        if perturbations is None:
            perturbations = dataset.get_unique_perturbations()
        if perturbations is not None:
            df = dataset.df.query(f"{dataset.perturbation_column} == @perturbations")
        else:
            df = dataset.df

        sns.scatterplot(
            data=df,
            x=dataset.features[0],
            y=dataset.features[1],
            style=dataset.perturbation_column,
            hue=dataset.perturbation_column,
            s=marker_size,
            markers=markers,
            ax=self.ax,
        )
        # self._decorate_figure()
        # plt.show()

    def _decorate_figure(self):
        """Internal helper function for axis decoration"""
        if self.config.get("show_legend", True):
            self.ax.legend()
        else:
            self.ax.get_legend().remove()

        self.ax.set_xlabel(self.config.get("xlabel", ""))
        self.ax.set_ylabel(self.config.get("ylabel", ""))
        axis_ranges = self.config.get("axis_ranges", ((None, None), (None, None)))
        self.ax.set_xlim(axis_ranges[0])
        self.ax.set_ylim(axis_ranges[1])
        self.ax.set_title(self.config.get("title", "2D scatter"))
        plt.tight_layout()

    def show(self):
        """Show the plot on the screen"""
        self._decorate_figure()
        plt.show()

    def save_figure(self, output_image_path: Union[Path, str], **savefig_kwargs):
        """Save the current scatter plot to PNG/SVG file

        Saving of the image is achieved using the matplotlib plt.savefig. After
        the output filename, any argument/option may be given that is also
        valid in calls to plt.savefig.

        Parameters
        ----------
        output_image_path : Union[Path, str]
            Output file path for the scatter plot.
        """
        self._decorate_figure()
        if isinstance(output_image_path, str):
            output_image_path = Path(output_image_path)
        if not output_image_path.parent.exists():
            output_image_path.parent.mkdir(parents=True)
        plt.savefig(output_image_path, **savefig_kwargs)

    def __del__(self):
        """Delete scatter

        Ensures plt.cla and clf are called upon Scatter object deletion
        """
        plt.cla()
        plt.clf()
