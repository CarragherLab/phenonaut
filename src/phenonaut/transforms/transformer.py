# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from collections.abc import Callable, Iterable
from copy import deepcopy
from inspect import isclass
from multiprocessing.sharedctypes import Value
from typing import List, Optional, Type, Union

import numpy as np
import pandas as pd
from numpy import array
from sklearn import decomposition

from phenonaut import data
from phenonaut.data.dataset import Dataset
from phenonaut.phenonaut import Phenonaut


class PheTransformerFitQueryMatchedNoRows(Exception):
    def __init__(self, message):
        super().__init__(message)


class Transformer:
    r"""Generic transformer to turn methods/objects into Phenonaut transforms

    This generic transformer class may be used to wrap functions which perform
    a simple transform (like np.log2), or even more complex objects, like PCA,
    t-SNE and StandardScaler from scikit (which provide their own fit/transform/
    fit_transform functions).

    This way, a Phenonaut transformer may be constructed with applies the given
    function/object method to a Phenonaut dataset and correctly updates features.

    Wrapping an object which requires fitting also brings with it the advantage
    that we may apply the groupby keyword to perform a unique fit and transform
    in groups, which is useful if you require PCA to be performed on a per-plate
    basis.

    May be used as follows - in this example, we are wrapping PCA from SciKit
    which has the effect of perfoming a 2D PCA on our Phenonaut Dataset object.
    We also make use of the groupby keyword, to perform a unique PCA for each
    plate (denoted by unique BARCODE values).


    .. code-block:: python

        from phenonaut import Phenonaut
        from phenonaut.transforms import Transformer
        import pandas as pd
        import numpy as np

        df=pd.DataFrame({
            'ROW':[1,1,1,1,1,1],
            'COLUMN':[1,1,1,2,2,2],
            'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
            'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
            'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
            'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
            'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
            'FOV':[1,2,1,2,1,2]})

        phe=Phenonaut(df)
        from sklearn.decomposition import PCA
        t_pca=Transformer(PCA, constructor_kwargs={'n_components':2})
        t_pca.fit(phe.ds, groupby="BARCODE")
        t_pca.transform(phe.ds)

    Along with the above, whereby a PCA transformer is generated from the SciKit
    PCA class, you may also use the built in phenonaut.transforms.PCA which has
    more PCA specific functionality, allowing creation of Scree plots etc.

    A transformer may also be made with a callable function (which operates on
    dataframes or numpy arrays) like np.log2, or np.abs, etc, as shown below.

    The following squares all features.

    .. code-block:: python

        from phenonaut import Phenonaut
        from phenonaut.transforms import Transformer
        import numpy as np
        import pandas as pd
        df=pd.DataFrame({
            'ROW':[1,1,1,1,1,1],
            'COLUMN':[1,1,1,2,2,2],
            'BARCODE':["Plate1","Plate1","Plate2","Plate2","Plate1","Plate1"],
            'feat_1':[1.2,1.3,5.2,6.2,0.1,0.2],
            'feat_2':[1.2,1.4,5.1,6.1,0.2,0.2],
            'feat_3':[1.3,1.5,5,6.8,0.3,0.38],
            'filename':['fileA.png','FileB.png','FileC.png','FileD.png','fileE.png','FileF.png'],
            'FOV':[1,2,1,2,1,2]})
        phe=Phenonaut(df)
        t=Transformer(np.square)
        t(phe)



    Parameters
    ----------
    method : Union[object, Callable]
        Instantiatable object with fit, transform and or fit_transform, or
        __call__ methods. Alternatively, a callable function. Designed to be
        passed something like the PCA class from SciKit, or a simple function
        like np.log2.
    new_feature_names : Optional[Union[str, list[str]]]
        List of strings containing the names for the new features. Can also
        be just a single string, which then has numerical suffixes attached
        enumerating the number of new features generated, however, if the
        string ends in an underscore, then the old feature name has that string
        prepended to it. For example, if 'StandardScaler\_' is given and the
        original features are feat_1, feat2, and feat_3, then the new features
        will be StandardScaler_feat_1, StandardScaler_feat_2, and
        StandardScaler_feat_3. If None, then the names of new features are
        attempted to be derived from the name of the wrapped function.
        By default None.
    transformer_name : Optional[str], optional
        When features are set on a Dataset, the history reflects what was
        carried out to generate those features. If None, then automatic naming
        of the passed function is attempted. By default None.
    constructor_kwargs : dict, optional
        Additional constructor arguments which may be passed to the class passed
        in the method argument upon instantiation. By default {}.
    callable_kwargs : dict, optional
        Additional arguments to pass to the function passed in the method
        argument. By default {}.
    fit_kwargs : dict, optional
        Additional arguments to the fit function called on object instantiated
        from the method argument. By default {}.
    transform_kwargs : dict, optional
        Additional arguments to the transform function called on object
        instantiated from the method argument. By default {}.
    fit_transform_kwargs : dict, optional
        Additional arguments to the fit_transform function called on object
        instantiated from the method argument. By default {}.
    """

    def __init__(
        self,
        method: Union[object, Callable],
        new_feature_names: Optional[Union[str, list[str]]] = None,
        transformer_name: Optional[str] = None,
        constructor_kwargs: dict = {},
        callable_kwargs: dict = {},
        fit_kwargs: dict = {},
        transform_kwargs: dict = {},
        fit_transform_kwargs: dict = {},
    ):
        self._fit_has_been_called = False
        self._original_method = method
        self._new_feature_names = new_feature_names
        self._constructor_kwargs = constructor_kwargs
        self._callable_kwargs = callable_kwargs
        self._fit_kwargs = fit_kwargs
        self._transform_kwargs = transform_kwargs
        self._fit_transform_kwargs = fit_transform_kwargs
        self._tranformer_name = transformer_name

        # Here we need to handle a method being put in, a class, and a class instance.
        if isclass(method):
            method = method(**constructor_kwargs)
        self._method = method

        self._has_fit = True if hasattr(self._method, "fit") else False
        self._has_transform = True if hasattr(self._method, "transform") else False
        self._has_fit_transform = (
            True if hasattr(self._method, "fit_transform") else False
        )
        self._is_callable = True if hasattr(self._method, "__call__") else False

    def fit(
        self,
        dataset: Union[Dataset, Phenonaut],
        groupby: Optional[Union[str, list[str]]] = None,
        fit_perturbation_ids: Union[str, list] = None,
        fit_query: Optional[str] = None,
        fit_kwargs: Optional[dict] = None,
    ):
        """Call fit on the transformer

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            Dataset containing data to be fitted against
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        fit_perturbation_ids : Union[str, list], optional
            If only a subset of the data should be used for fitting, then
            pertubations for fitting may be listed here. If none, then
            every datapoint is used for fitting, by default None.
        fit_query : Optional[str], optional
            A pandas style query may be supplied to perform fitting. By default
            None.
        fit_kwargs : Optional[dict], optional
            Optional arguments supplied to the fit function of the object passed
            in the constructor of this transformer. By default None.

        """
        if fit_kwargs is None:
            fit_kwargs = self._fit_kwargs
        if fit_perturbation_ids is not None and fit_query is not None:
            raise ValueError(
                "Cannot supply both fit_perturbation_ids and fit_query together, only one can be used"
            )
        if not isinstance(dataset, (Phenonaut, Dataset)):
            raise TypeError(
                "Supplied data must be of type phenonaut.Dataset, or a Phenonaut object, "
                "in which case behavior is as if .ds was called to return the dataset with"
                f" the highest index (last added), found type was {type(dataset)}"
            )

        if isinstance(dataset, Phenonaut):
            dataset = dataset.ds

        if fit_perturbation_ids is not None:
            if isinstance(fit_perturbation_ids, str):
                fit_perturbation_ids = [fit_perturbation_ids]
            fit_perturbation_ids = list(fit_perturbation_ids)
            fit_query = f"{dataset.perturbation_column} == @fit_perturbation_ids"

        if groupby is None:
            if fit_query is None:
                self._method.fit(dataset.data, **fit_kwargs)
            else:
                query_res = dataset.df.query(fit_query)
                if len(query_res) == 0:
                    raise PheTransformerFitQueryMatchedNoRows(
                        f"Nothing to fit to, '{fit_query}' returned an empty DataFrame"
                    )
                self._method.fit(query_res[dataset.features], **fit_kwargs)
        else:
            self._original_features = dataset.features
            self._grouped_data = dataset.df.groupby(groupby)
            self._dataframes = [gdf for _, gdf in self._grouped_data]
            self._method = []
            for df in self._dataframes:
                if isclass(self._original_method):
                    self._method.append(
                        self._original_method(**self._constructor_kwargs)
                    )
                else:
                    self._method.append(deepcopy(self._orignal_method))
                if fit_query is None:
                    self._method[-1].fit(df[self._original_features], **fit_kwargs)
                else:
                    self._method[-1].fit(
                        df.query(fit_query)[self._original_features], **fit_kwargs
                    )
        self._fit_has_been_called = True
        return self

    def transform(
        self,
        dataset: Union[Dataset, Phenonaut],
        new_feature_names: Optional[Union[list[str], str]] = None,
        method_name: Optional[str] = None,
        transform_kwargs: Optional[dict] = None,
        free_memory_after_transform: bool = True,
        center_on_perturbation_id: Optional[str] = False,
        centering_function: Callable = np.median,
    ):
        """Apply transform

        If no transform method is found, then __call__ is called.

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object upon which the transform
            should be applied.
        new_feature_names : Optional[Union[list[str], str]], optional
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. If None, then
            features are attempted to be named through interrogation of the
            method_name argument. By default None.
        method_name : Optional[str], optional
            When setting features, the history message should describe what was
            done. If None, then the method object/method is interrogated in an
            attempt to automatically deduce the name. By default None.
        transform_kwargs : Optional[dict], optional
            Additional arguments to the transform function called on object
            instantiated from the method argument. If set here, and not None, then
            it overrides any transform_kwargs given to the object constructor. By
            default None.
        free_memory_after_transform : bool, optional
            Remove temporary objects once transformation has been performed.
            When performing a groupby operation, intermediate fits are retained,
            these may be deleted after calling transform by setting this argument
            to True. By default True.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the PCA on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable, optional
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.

        """
        if not self._has_transform:
            if self._is_callable:
                self.__call__(
                    dataset,
                    new_feature_names=new_feature_names,
                    method_name=method_name,
                )
            else:
                raise ValueError(
                    "No transform method found, and transform is also not callable."
                )
        if transform_kwargs is None:
            transform_kwargs = self._transform_kwargs
        if not isinstance(dataset, (Phenonaut, Dataset)):
            raise TypeError(
                "Supplied data must be of type phenonaut.Dataset, or a Phenonaut object, "
                "in which case behavior is as if .ds was called to return the dataset with"
                f" the highest index (last added), found type was {type(dataset)}"
            )
        if isinstance(dataset, Phenonaut):
            dataset = dataset.ds

        if isinstance(self._method, list):
            if len(self._method) == 0 or len(self._method) != len(self._dataframes):
                raise ValueError(
                    f"groupby appears to have failed, there are {len(self._method)} transformers, and {len(self._dataframes)} groups"
                )

            for df_index, (fitted_object, df) in enumerate(
                zip(self._method, self._dataframes)
            ):
                transformed_data_array = np.array(
                    fitted_object.transform(
                        df[self._original_features], **transform_kwargs
                    )
                )
                (
                    method_name,
                    new_feature_names,
                ) = self._get_method_name_and_feature_names(
                    transformed_data_array.shape[1],
                    method_name,
                    dataset.features,
                    new_feature_names,
                    dataset.df.columns.values.tolist(),
                )

                transformed_df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            data=transformed_data_array,
                            columns=new_feature_names,
                            index=df.index,
                        ),
                    ],
                    axis=1,
                )

                self._dataframes[df_index] = self._center_df(
                    dataset,
                    transformed_df,
                    center_on_perturbation_id,
                    new_feature_names,
                    centering_function,
                )

            dataset.df = pd.concat(self._dataframes, axis=0, ignore_index=True)
            if free_memory_after_transform:
                del self._dataframes
                del transformed_data_array
                self._method = (
                    self._original_method(**self._constructor_kwargs)
                    if isclass(self._original_method)
                    else self._original_method
                )
        else:
            transformed_data_array = np.array(
                self._method.transform(dataset.data, **transform_kwargs)
            )

            method_name, new_feature_names = self._get_method_name_and_feature_names(
                transformed_data_array.shape[1],
                method_name,
                dataset.features,
                new_feature_names,
                dataset.df.columns.values.tolist(),
            )
            transformed_df = pd.concat(
                [
                    dataset.df,
                    pd.DataFrame(
                        data=transformed_data_array,
                        columns=new_feature_names,
                        index=dataset.df.index,
                    ),
                ],
                axis=1,
            )
            # Attempt to center on perturbation ids,
            dataset.df = self._center_df(
                dataset,
                transformed_df,
                center_on_perturbation_id,
                new_feature_names,
                centering_function,
            )

        dataset.features = (
            new_feature_names,
            f"Applied {method_name}, {self._constructor_kwargs=}, {self._fit_kwargs=}, {self._transform_kwargs=}, {self._fit_transform_kwargs=}, {self._callable_kwargs=}, centered on {center_on_perturbation_id}, {centering_function}",
        )

    def _center_df(
        self,
        dataset,
        df,
        center_on_perturbation_id,
        new_feature_names,
        centering_function,
    ):
        if center_on_perturbation_id is not None:
            if isinstance(center_on_perturbation_id, str):
                center_on_coords = centering_function(
                    df.query(
                        f"{dataset.perturbation_column} =='{center_on_perturbation_id}'"
                    )[new_feature_names],
                    axis=0,
                )
            else:
                center_on_coords = center_on_perturbation_id
            df.loc[:, new_feature_names] = (
                df.loc[:, new_feature_names] - center_on_coords
            )
        return df

    def fit_transform(
        self,
        dataset: Union[Dataset, Phenonaut],
        groupby: Optional[Union[str, list[str]]] = None,
        fit_perturbation_ids: Union[str, list] = None,
        fit_query: Optional[str] = None,
        fit_kwargs: Optional[dict] = None,
        transform_kwargs: Optional[dict] = None,
        fit_transform_kwargs: Optional[dict] = None,
        new_feature_names: Optional[Union[str, list[str]]] = None,
        method_name: Optional[str] = None,
        free_memory_after_transform: bool = True,
        center_on_perturbation_id: Optional[str] = False,
        centering_function: Callable = np.mean,
    ):
        """_summary_

        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object upon which the
            fit_transform should be applied.
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        fit_perturbation_ids : Union[str, list], optional
            If only a subset of the data should be used for fitting, then
            pertubations for fitting may be listed here. If none, then
            every datapoint is used for fitting, by default None.
        fit_query : Optional[str], optional
            A pandas style query may be supplied to perform fitting. By default
            None.
        fit_kwargs : Optional[dict], optional
            Optional arguments supplied to the fit function of the object passed
            in the constructor of this transformer. By default None.
        transform_kwargs : Optional[dict], optional
            Additional arguments to the transform function called on object
            instantiated from the method argument. If set here, and not None, then
            it overrides any transform_kwargs given to the object constructor.
            Transform is only called as a fallback, if the object does not have
            a fit_transform method, but separate fit and transform methods which
            may be called in series. By default None.
        fit_transform_kwargs : Optional[dict], optional
            Additional arguments to the fit_transform function called on object
            instantiated from the method argument. If set here, and not None, then
            it overrides any fit_transform_kwargs given to the object constructor.
            By default None.
        new_feature_names : Optional[Union[list[str], str]], optional
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. If None, then
            features are attempted to be named through interrogation of the
            method_name argument. By default None.
        method_name : Optional[str], optional
            When setting features, the history message should describe what was
            done. If None, then the method object/method is interrogated in an
            attempt to automatically deduce the name. By default None.
        free_memory_after_transform : bool, optional
            Remove temporary objects once transformation has been performed.
            When performing a groupby operation, intermediate fits are retained,
            these may be deleted after calling transform by setting this argument
            to True. By default True.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the PCA on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable, optional
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.

        """
        if not isinstance(dataset, (Phenonaut, Dataset)):
            raise TypeError(
                f"Dataset should be a Phenonaut object, or phenonaut.Dataset object, it was {type(dataset)}"
            )
        if isinstance(dataset, Phenonaut):
            dataset = dataset.ds

        if fit_kwargs is None:
            fit_kwargs = self._fit_kwargs
        if transform_kwargs is None:
            transform_kwargs = self._fit_kwargs
        if fit_transform_kwargs is None:
            fit_transform_kwargs = self._fit_transform_kwargs

        if not self._has_fit_transform:
            self.fit(
                dataset,
                groupby=groupby,
                fit_perturbation_ids=fit_perturbation_ids,
                fit_query=fit_query,
                fit_kwargs=fit_kwargs,
            )
            self.transform(
                dataset,
                new_feature_names=new_feature_names,
                transform_kwargs=transform_kwargs,
                free_memory_after_transform=free_memory_after_transform,
                center_on_perturbation_id=center_on_perturbation_id,
                centering_function=centering_function,
            )
        else:
            if fit_perturbation_ids is not None or fit_query is not None:
                raise ValueError(
                    f"fit_perturbation_ids and fit_query cannot be used when the transformation has a native fit_transform method.  Call fit, then transform on the transformer"
                )
            if groupby is None:
                # Below we check if fit_transform_kwargs has anythin in it. This is not needed
                # for scikit and similar, but the UMAP package thinks that a y argument of length 0
                # if being passed if anything is there, so we work around it.
                if fit_transform_kwargs is None or fit_transform_kwargs == {}:
                    transformed_data_array = np.array(
                        self._method.fit_transform(dataset.data)
                    )
                else:
                    transformed_data_array = np.array(
                        self._method.fit_transform(dataset.data, **fit_transform_kwargs)
                    )
                (
                    method_name,
                    new_feature_names,
                ) = self._get_method_name_and_feature_names(
                    transformed_data_array.shape[1],
                    method_name,
                    dataset.features,
                    new_feature_names,
                    dataset.df.columns.values.tolist(),
                )
                transformed_df = pd.concat(
                    [
                        dataset.df,
                        pd.DataFrame(
                            data=transformed_data_array,
                            columns=new_feature_names,
                            index=dataset.df.index,
                        ),
                    ],
                    axis=1,
                )
                dataset.df = self._center_df(
                    dataset,
                    transformed_df,
                    center_on_perturbation_id,
                    new_feature_names,
                    centering_function,
                )

            else:  # Groupby is used
                self._original_features = dataset.features
                self._grouped_data = dataset.df.groupby(groupby)
                self._dataframes = [gdf for _, gdf in self._grouped_data]
                self._method = []
                for df_index, df in enumerate(self._dataframes):
                    if isclass(self._original_method):
                        self._method.append(
                            self._original_method(**self._constructor_kwargs)
                        )
                    else:
                        self._method.append(deepcopy(self._orignal_method))
                    transformed_data_array = np.array(
                        self._method[-1].fit_transform(
                            df[dataset.features], **fit_transform_kwargs
                        )
                    )

                    (
                        method_name,
                        new_feature_names,
                    ) = self._get_method_name_and_feature_names(
                        transformed_data_array.shape[1],
                        method_name,
                        dataset.features,
                        new_feature_names,
                        dataset.df.columns.values.tolist(),
                    )

                    transformed_df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                data=transformed_data_array,
                                columns=new_feature_names,
                                index=df.index,
                            ),
                        ],
                        axis=1,
                    )
                    self._dataframes[df_index] = self._center_df(
                        dataset,
                        transformed_df,
                        center_on_perturbation_id,
                        new_feature_names,
                        centering_function,
                    )
                dataset.df = pd.concat(self._dataframes, axis=0, ignore_index=True)
                if free_memory_after_transform:
                    del self._dataframes
            dataset.features = (
                new_feature_names,
                f"Applied {method_name}, {self._constructor_kwargs=}, {self._fit_kwargs=}, {self._transform_kwargs=}, {self._fit_transform_kwargs=}, {self._callable_kwargs=}",
            )
            if free_memory_after_transform:
                del transformed_data_array

    def _get_method_name_and_feature_names(
        self,
        n_features: int,
        method_name: Optional[str] = None,
        old_features: Optional[list[str]] = None,
        new_features: Optional[list[str]] = None,
        existing_column_names: Optional[list[str]] = None,
    ):
        """Get method name and new feature names

        Parameters
        ----------
        n_features : int
            Number of new features to generate
        method_name : Optional[str], optional
            Name of the method, if known. If supplied and not None, then this
            name will be used. By default None.
        new_features : Optional[list[str]], optional
            If the new features are known, then these are returned instead, by default None
        existing_column_names : Optional[list[str]]
            If supplied, then the function performs a check to see that any automatically
            generated feature names are not already present in the dataframe columns. If they
            are, then "_1" is appended. This check is performed iteratively, increasing the
            number at each iteration until a set of unused features are found.

        Returns
        -------
        tuple(str, list[str])
            Tuple containing the method name and new feature names list.
        """
        if method_name is None:
            if isinstance(self._method, list):
                m = self._method[-1]
            else:
                m = self._method
            if hasattr(m, "__name__"):
                method_name = m.__name__
            elif hasattr(m, "__class__"):
                method_name = m.__class__.__name__
            else:
                method_name = "UnknownTransform"

        if new_features is None:
            new_features = self._new_feature_names
        if new_features is None:
            new_features = method_name
        if isinstance(new_features, str):
            if (
                new_features[-1] == "_"
                and isinstance(old_features, list)
                and len(old_features) == n_features
            ):
                new_features = [f"{new_features}{f}" for f in old_features]
            else:
                new_features = [f"{new_features}_{i+1}" for i in range(n_features)]

        if existing_column_names is not None:
            f_repeat_counter = 1
            if set(new_features).intersection(set(existing_column_names)):
                new_features = [f"{f}_{f_repeat_counter}" for f in new_features]
            while set(new_features).intersection(set(existing_column_names)):
                f_repeat_counter += 1
                new_features = [
                    f"{'_'.join(f.split('_')[:-1])}_{f_repeat_counter}"
                    for f in new_features
                ]
        return method_name, new_features

    def __call__(
        self,
        dataset: Union[Dataset, Phenonaut, np.ndarray],
        groupby: Optional[Union[str, list[str]]] = None,
        fit_perturbation_ids: Optional[Union[str, list]] = None,
        fit_query: Optional[str] = None,
        fit_kwargs: Optional[dict] = None,
        transform_kwargs: Optional[dict] = None,
        fit_transform_kwargs: Optional[dict] = None,
        new_feature_names: Optional[Union[str, list[str]]] = None,
        method_name: Optional[str] = None,
        free_memory_after_transform: bool = True,
        center_on_perturbation_id: Optional[str] = False,
        centering_function: Callable = np.median,
    ):
        """Call transformer

        If a simple callable method was passed to the constructor of the transformer,
        then it can be applied by calling the method here.  The transform method can
        also be called with the same effect.  If the method has no __call__ method,
        then transform is attempted to be called. If this is absent, then fit_transform
        is attempted.


        Parameters
        ----------
        dataset : Union[Dataset, Phenonaut]
            The Phenonaut dataset or Phenonaut object upon which the transform
            should be applied.
        groupby : Optional[Union[str, list[str]]], optional
            Often we would like to apply transformations on a plate-by-plate
            basis. This groupby argument allows definition of a string which
            is used to identify columns with which unique values define a group
            or plate. It works with the pandas groupby function and accepts
            a list of strings if multiple columns must be used to define groups.
            By default None.
        fit_perturbation_ids : Union[str, list], optional
            If only a subset of the data should be used for fitting, then
            pertubations for fitting may be listed here. If none, then
            every datapoint is used for fitting, by default None.
        fit_query : Optional[str], optional
            A pandas style query may be supplied to perform fitting. By default
            None.
        fit_kwargs : Optional[dict], optional
            Optional arguments supplied to the fit function of the object passed
            in the constructor of this transformer. By default None.
        transform_kwargs : Optional[dict], optional
            Additional arguments to the transform function called on object
            instantiated from the method argument. If set here, and not None, then
            it overrides any transform_kwargs given to the object constructor.
            Transform is only called as a fallback, if the object does not have
            a fit_transform method, but separate fit and transform methods which
            may be called in series. By default None.
        fit_transform_kwargs : Optional[dict], optional
            Additional arguments to the fit_transform function called on object
            instantiated from the method argument. If set here, and not None, then
            it overrides any fit_transform_kwargs given to the object constructor.
            By default None.
        new_feature_names : Optional[Union[list[str], str]], optional
            List of strings containing the names for the new features. Can also
            be just a single string, which then has numerical suffixes attached
            enumerating the number of new features generated. If None, then
            features are attempted to be named through interrogation of the
            method_name argument. By default None.
        method_name : Optional[str], optional
            When setting features, the history message should describe what was
            done. If None, then the method object/method is interrogated in an
            attempt to automatically deduce the name. By default None.
        free_memory_after_transform : bool, optional
            Remove temporary objects once transformation has been performed.
            When performing a groupby operation, intermediate fits are retained,
            these may be deleted after calling transform by setting this argument
            to True. By default True.
        center_on_perturbation_id : Optional[str], optional
            Optionally, recentre the PCA on a named perturbation. Should have
            pertubation_column set within the dataset for this option, by
            default None.
        centering_function : Callable, optional
            Used with center_on_perturbation, this function is applied to
            all data for matching perturbations. By default, we use the median
            of perturbations. This behavior can be overridden by supplying a
            different function here. By default np.median.
        """
        if not isinstance(dataset, (Phenonaut, Dataset)):
            raise TypeError(
                f"Dataset should be a Phenonaut object, or phenonaut.Dataset object, it was {type(dataset)}"
            )
        if isinstance(dataset, Phenonaut):
            dataset = dataset.ds

        if self._is_callable:
            if self._callable_kwargs == {} or self._callable_kwargs is None:
                transformed_data_array = np.array(self._method(dataset.data))
            else:
                transformed_data_array = np.array(
                    self._method(dataset.data, **self._callable_kwargs)
                )

            method_name, new_feature_names = self._get_method_name_and_feature_names(
                transformed_data_array.shape[1],
                method_name,
                dataset.features,
                new_feature_names,
                dataset.df.columns.values.tolist(),
            )
            transformed_df = pd.DataFrame(
                data=transformed_data_array,
                columns=new_feature_names,
                index=dataset.df.index,
            )
            dataset.df = pd.concat([dataset.df, transformed_df], axis=1)
            dataset.features = (
                new_feature_names,
                f"Applied callable {method_name}, {self._callable_kwargs=}",
            )
            return
        else:
            # Not callable
            if self._fit_has_been_called:
                if self._has_transform:
                    self.transform(
                        dataset,
                        new_feature_names=new_feature_names,
                        method_name=method_name,
                        transform_kwargs=transform_kwargs,
                        free_memory_after_transform=free_memory_after_transform,
                        center_on_perturbation_id=center_on_perturbation_id,
                        centering_function=centering_function,
                    )
            else:
                self.fit_transform(
                    dataset,
                    fit_perturbation_ids=fit_perturbation_ids,
                    fit_query=fit_query,
                    fit_kwargs=fit_kwargs,
                    transform_kwargs=transform_kwargs,
                    fit_transform_kwargs=fit_transform_kwargs,
                    new_feature_names=new_feature_names,
                    method_name=method_name,
                    free_memory_after_transform=free_memory_after_transform,
                    center_on_perturbation_id=center_on_perturbation_id,
                    centering_function=centering_function,
                )
