# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from numpy import isin
from pandas.errors import DataError
from sklearn.decomposition import PCA as _SKLearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearn_LDA
from sklearn.manifold import TSNE as _SKLearn_TSNE

from phenonaut import data
from phenonaut.data import Dataset
from phenonaut.phenonaut import Phenonaut
from phenonaut.transforms.transformer import Transformer


class PCA(Transformer):
    """Principal Component Analysis (PCA) dimensionality reduction.

    Can be instantiated and called like:

    .. code-block:: python

        pca=PCA()
        pca(dataset)

    PCA transformer is rather verbose, copying a lot of functionality supplied by
    its parent class simply becuase explaining feature variance is often important
    along with generating skree plots etc.

    Parameters
    ----------
    dataset : Dataset
        The Phenonaut dataset on which to apply PCA
    new_feature_names : Union[list[str], str]
        List of strings containing the names for the new features. Can also
        be just a single string, which then has numerical suffixes attached
        enumerating the number of new features generated. By default "PC"
    ndims : int, optional
        Number of dimensions to embed the data into, by default 2
    """

    def __init__(self, new_feature_names="PC", ndims: int = 2):
        super().__init__(
            _SKLearnPCA,
            new_feature_names=new_feature_names,
            transformer_name="SciKitPCA",
            constructor_kwargs={"n_components": ndims},
        )
        self.ndims = ndims

    def __call__(
        self,
        dataset: Union[Dataset, Phenonaut],
        ndims: Optional[int] = None,
        new_feature_names: Union[list[str], str] = "PC",
        groupby: Optional[Union[str, list[str]]] = None,
        center_on_perturbation_id: Optional[str] = None,
        centering_function: Callable = np.median,
        fit_perturbation_ids: Optional[Union[str, list]] = None,
        fit_query: Optional[str] = None,
        explain_variance_in_features: bool = False,
    ):
        """Principal Component Analysis (PCA) dimensionality reduction

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object containing one dataset
            on which to apply PCA.
        ndims : int
            Number of dimensions to embed the data into. If None, then the value of
            ndims passed to the constructor is used.
        new_feature_names : Union[list[str], str]
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. By default "PC"
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the PCA on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.
        fit_perturbation_ids : Optional[Union[str, list]], optional
            If only a subset of the data should be used for fitting, then
            pertubations for fitting may be listed here. If none, then
            every datapoint is used for fitting, by default None.
        fit_query : str, optional
            A pandas style query may be supplied to perform fitting. By default
            None.
        explain_variance_in_features : bool, optional
            If True, then the percentage explained variance of each PCA
            dimension is included in the new PCA descriptor feature name.
            Overrides new_feature_names. By default False.
        """
        if ndims == None:
            ndims = self.ndims
        if self.ndims != ndims:
            super().__init__(
                _SKLearnPCA,
                new_feature_names=new_feature_names,
                transformer_name="SciKitPCA",
                constructor_kwargs={"n_components": ndims},
            )
            self.ndims = ndims
        self.fit_transform(
            dataset,
            groupby=groupby,
            fit_perturbation_ids=fit_perturbation_ids,
            fit_query=fit_query,
            new_feature_names=new_feature_names,
            method_name="SciKitPCA",
            center_on_perturbation_id=center_on_perturbation_id,
            centering_function=centering_function,
        )
        if explain_variance_in_features:
            if isinstance(self._method, list):
                print(
                    "Warning, adding variance explained by features, however, groupby was used. Explaining variance using the first fitted PCA"
                )
                pca_variance_ratio = self._method[0].explained_variance_ratio_
            else:
                pca_variance_ratio = self._method.explained_variance_ratio_

            explained_var_features = [
                f"PC{x+1} ({expvariance*100:.2f} % explained variance)"
                for x, expvariance in zip(range(self.ndims), pca_variance_ratio)
            ]

            column_rename_dict = {
                k: v for k, v in zip(dataset.features, explained_var_features)
            }
            dataset.df.rename(columns=column_rename_dict, inplace=True)
            dataset.features = (
                explained_var_features,
                "Added explained variance to features",
            )

    def save_scree_plot(
        self, output_filename: Optional[Union[str, Path]] = None, title="Scree plot"
    ):
        """Produce a Scree plot showing ndims vs explained variance

        Parameters
        ----------
        output_filename : Optional[Union[str, Path]]
            Output filename (ending in .png) indicating where the Scree plot should
            be saved. If None, then the plot is displayed interactively. By default, None.
        title : str, optional
            Plot title, by default "Scree plot"

        Raises
        ------
        DataError
            PCA must have been fitted before a Scree plot can be made.
        """
        if self._method is None:
            raise DataError("Cant make scree plot, nothing fitted")
        if isinstance(self._method, list):
            print(
                "Multiple fitted PCAs found (groupby used?) producing Scree plot for the last"
            )
            pca_object = self._method[-1]
        else:
            pca_object = self._method
        fig, ax = plt.subplots(1)
        ax.plot(
            range(1, pca_object.explained_variance_ratio_.shape[0] + 1),
            np.cumsum(pca_object.explained_variance_ratio_),
            label="Cumulative explained variance",
            marker="x",
        )
        ax.plot(
            range(1, pca_object.explained_variance_ratio_.shape[0] + 1),
            pca_object.explained_variance_ratio_,
            label="Explained variance",
            marker="x",
        )

        ax.set_xlabel("Feature")
        ax.set_ylabel("Fraction explained variance")
        ax.set_xlim((0.25, pca_object.explained_variance_ratio_.shape[0]))
        ax.set_ylim((0, 1))
        ax.set_title(title)
        plt.xticks(range(1, pca_object.explained_variance_ratio_.shape[0] + 1))
        plt.tight_layout()
        plt.grid()
        ax.legend()
        if output_filename is None:
            plt.show()
        else:
            if isinstance(output_filename, str):
                output_image_path = Path(output_filename)
            plt.savefig(output_filename)


class TSNE(Transformer):
    """t-SNE dimensionality reduction

    Can be instantiated and called like:

    .. code-block:: python

        tsne=TSNE()
        tsne(dataset)

    Parameters
    ----------
    dataset : Dataset
        The Phenonaut dataset on which to apply the transformation
    new_feature_names : Union[list[str], str]
        List of strings containing the names for the new features. Can also
        be just a single string, which then has numerical suffixes attached
        enumerating the number of new features generated. By default "TSNE"
    ndims : int, optional
        Number of dimensions to embed the data into, by default 2
    """

    def __init__(self, constructor_kwargs={}, new_feature_names="TSNE", ndims: int = 2):
        constructor_kwargs["n_components"] = ndims

        if ndims > 3:
            constructor_kwargs["method"] = "exact"
        else:
            constructor_kwargs["method"] = "barnes_hut"

        super().__init__(
            _SKLearn_TSNE,
            new_feature_names=new_feature_names,
            transformer_name="t-SNE",
            constructor_kwargs=constructor_kwargs,
        )
        self.ndims = ndims

    def __call__(
        self,
        dataset: Union[Dataset, Phenonaut],
        ndims: int = 2,
        new_feature_names: Union[list[str], str] = "tSNE",
        groupby: Optional[Union[str, list[str]]] = None,
        center_on_perturbation_id: Optional[str] = None,
        centering_function: Callable = np.median,
    ):
        """t-SNE dimensionality reduction

        Once instantiated, can be called directly, like:

        .. code-block:: python

            tsne=TSNE()
            tsne(dataset)

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object containing one dataset
            on which to apply t-SNE.
        ndims : int
            Number of dimensions to embed the data into, by default 2
        new_feature_names : Union[list[str], str]
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. By default "tSNE"
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the t-SNE on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.
        """
        if self.ndims != ndims:
            constructor_kwargs = {"n_components": ndims}
            if ndims > 3:
                constructor_kwargs["method"] = "exact"
            else:
                constructor_kwargs["method"] = "barnes_hut"
            super().__init__(
                _SKLearn_TSNE,
                new_feature_names=new_feature_names,
                transformer_name="t-SNE",
                constructor_kwargs=constructor_kwargs,
            )
            self.ndims = ndims

        self.fit_transform(
            dataset,
            groupby=groupby,
            new_feature_names=new_feature_names,
            method_name="t-SNE",
            center_on_perturbation_id=center_on_perturbation_id,
            centering_function=centering_function,
        )


class UMAP(Transformer):
    """UMAP dimensionality reduction

    Can be instantiated and called like:

    .. code-block:: python

        umap=UMAP()
        umap(dataset)

    Parameters
    ----------
    dataset : Dataset
        The Phenonaut dataset on which to apply the transformation
    new_feature_names : Union[list[str], str]
        List of strings containing the names for the new features. Can also
        be just a single string, which then has numerical suffixes attached
        enumerating the number of new features generated. By default "UMAP"
    ndims : int, optional
        Number of dimensions to embed the data into, by default 2
    umap_kwargs : dict
        Keyword arguments to pass to UMAP-lean constructor. Often the number of
        neighbors requires changing and this can be achieved here by passing in
        {'n_neighbors': 50} for example, to run UMAP with 50 neighbors.  Any value of
        an n_components key within this dictionary will be overwritten by the value of
        the ndims argument.
    """

    def __init__(
        self, new_feature_names="UMAP", ndims: int = 2, umap_kwargs: dict = {}
    ):
        self.umap_object_kwargs = {"n_components": ndims}
        self.umap_object_kwargs.update(umap_kwargs)
        super().__init__(
            umap.UMAP,
            new_feature_names=new_feature_names,
            transformer_name="UMAP",
            constructor_kwargs=self.umap_object_kwargs,
        )
        self.ndims = ndims

    def __call__(
        self,
        dataset: Union[Dataset, Phenonaut],
        ndims: int = 2,
        new_feature_names: Union[list[str], str] = "UMAP",
        groupby: Optional[Union[str, list[str]]] = None,
        center_on_perturbation_id: Optional[str] = None,
        centering_function: Callable = np.median,
    ):
        """UMAP dimensionality reduction

        Once instantiated, can be called directly, like:

        .. code-block:: python

            uamp=UMAP()
            umap(dataset)

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object containing one dataset
            on which to apply UMAP.
        ndims : int
            Number of dimensions to embed the data into, by default 2
        new_feature_names : Union[list[str], str]
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. By default "UMAP"
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the UMAP on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.
        """
        if self.ndims != ndims:
            self.umap_object_kwargs.update({"n_components": ndims})
            super().__init__(
                umap.UMAP,
                new_feature_names=new_feature_names,
                transformer_name="UMAP",
                constructor_kwargs=self.umap_object_kwargs,
            )
            self.ndims = ndims

        self.fit_transform(
            dataset,
            groupby=groupby,
            new_feature_names=new_feature_names,
            method_name="UMAP",
            center_on_perturbation_id=center_on_perturbation_id,
            centering_function=centering_function,
        )


class LDA:
    """LDA dimensionality reduction

    Once instantiated, can be called like:

    .. code-block:: python

        lda=LDA()
        lda(dataset)


    Parameters
    ----------
    dataset : Union[Dataset, Phenonaut]
        The Phenonaut dataset or Phenonaut object containing one dataset
        on which to apply the transformation.
    ndims : int, optional
        Number of dimensions to embed the data into, by default 2
    center_on_perturbation_id : Optional[str], optional
        Optionally, recentre the embedding space on a named perturbation. Should have
        pertubation_column set within the dataset for this option, by
        default None.
    center_by_median : bool, optional
        By default, any dataset centering will be performed on the median
        of controls or perturbations. If this argument is False, then
        centering is performed on the mean, by default True.
    predict_proba : bool
        If True, then probabilities of each datapoint belonging to every
        other class is calculated and used in place of output features.
    """

    def __init__(self):
        self.ndims = 2
        self.lda = None

    def __call__(
        self,
        dataset: Dataset,
        ndims=2,
        center_on_perturbation_id=None,
        center_by_median: bool = True,
        predict_proba: bool = False,
    ):
        """LDA dimensionality reduction

        Once instantiated, can be called like:

        .. code-block:: python

            lda=LDA()
            lda(dataset)


        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object containing one dataset
            on which to apply the transformation.
        ndims : int, optional
            Number of dimensions to embed the data into, by default 2
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the embedding space on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        center_by_median : bool, optional
            By default, any dataset centering will be performed on the median
            of controls or perturbations. If this argument is False, then
            centering is performed on the mean, by default True.
        predict_proba : bool
            If True, then probabilities of each datapoint belonging to every
            other class is calculated and used in place of output features.
        """
        if isinstance(dataset, Phenonaut):
            if len(dataset.datasets) == 1:
                dataset = dataset.ds
            else:
                raise ValueError(
                    "Phenonaut object with more than one dataset passed to fit, please pass the dataframe to apply the transform to it"
                )
        lda_feature_column_names = [f"LDA{x+1}" for x in range(ndims)]
        fit_perturbation_ids = list(dataset.df[dataset.perturbation_column].unique())

        self.lda = sklearn_LDA(n_components=ndims)
        X, y = (
            dataset.df.loc[:, dataset.features],
            dataset.df.loc[:, dataset.perturbation_column],
        )
        self.lda.fit(X, y)

        dataset.df.loc[:, lda_feature_column_names] = self.lda.transform(X)
        dataset.features = (lda_feature_column_names, f"Performed LDA, ndims={ndims}")
        if center_on_perturbation_id is not None:
            if isinstance(center_on_perturbation_id, str):
                if center_by_median:
                    center_on_coords = np.median(
                        dataset.df.query(
                            f"{dataset.perturbation_column} =='{center_on_perturbation_id}'"
                        )[dataset.features],
                        axis=0,
                    )
                else:
                    center_on_coords = np.mean(
                        dataset.df.query(
                            f"{dataset.perturbation_column} =='{center_on_perturbation_id}'"
                        )[dataset.features],
                        axis=0,
                    )
            else:
                center_on_coords = center_on_perturbation_id
            dataset.df.loc[:, dataset.features] = (
                dataset.df.loc[:, dataset.features] - center_on_coords
            )
            dataset.features = (
                dataset.features,
                f"Centered on {center_on_perturbation_id}, {center_by_median=}",
            )

        if predict_proba:
            dataset.df[
                [f"proba_{c}" for c in self.lda.classes_]
            ] = self.lda.predict_proba(dataset.df[dataset.features])
            for index, row in dataset.df.iterrows():
                correct_proba = row[f"proba_{row['Edit']}"]
                dataset.df.loc[index, "Correct_proba"] = correct_proba

            dataset.features = (
                [f"proba_{c}" for c in self.lda.classes_],
                "LDA prediction probability",
            )
            dataset.df.to_csv("a.csv")
        else:
            dataset.features = (
                lda_feature_column_names,
                f"Performed LDA, ndims={ndims}",
            )

    def make_scree_plot(self, output_filename: None, title="Scree plot"):
        """Produce a Scree plot showing ndims vs explained variance

        Parameters
        ----------
        output_filename : Optional[Union[str, Path]]
            Output filename (ending in .png) indicating where the Scree plot should
            be saved. If None, then the plot is displayed interactively. By default, None.
        title : str, optional
            Plot title, by default "Scree plot"

        Raises
        ------
        DataError
            LDA must have been fitted before a Scree plot can be made.
        """
        if self.lda is None:
            raise DataError("Cant make scree plot, nothing fitted")
        fig, ax = plt.subplots(1)
        ax.plot(
            range(1, self.lda.explained_variance_ratio_.shape[0] + 1),
            np.cumsum(self.lda.explained_variance_ratio_),
            label="Cumulative explained variance",
            marker="x",
        )
        ax.plot(
            range(1, self.lda.explained_variance_ratio_.shape[0] + 1),
            self.lda.explained_variance_ratio_,
            label="Explained variance",
            marker="x",
        )

        ax.set_xlabel("Feature")
        ax.set_ylabel("Fraction explained variance")

        ax.set_xlim((0.25, self.lda.explained_variance_ratio_.shape[0]))
        ax.set_ylim((0, 1))
        ax.set_title(title)
        plt.xticks(range(1, self.lda.explained_variance_ratio_.shape[0] + 1))
        plt.tight_layout()
        plt.grid()
        ax.legend()
        if output_filename is None:
            plt.show()
        else:
            if isinstance(output_filename, str):
                output_image_path = Path(output_filename)
            plt.savefig(output_filename)
